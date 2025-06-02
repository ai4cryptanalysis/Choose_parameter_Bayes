import os
NUM=0#
os.environ["CUDA_VISIBLE_DEVICES"] = str(NUM)#
import gc
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Dense, Conv1D,Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from pickle import dump
import tensorflow as tf
import simeck as sk
import numpy as np
import concurrent.futures

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')#前面指定了使用的显卡后，这里存在的显卡就只有一张了
tf.config.experimental.set_memory_growth(physical_devices[0], True)


wdir = './good_trained_nets/'
word_size = sk.WORD_SIZE()

if(not os.path.exists(wdir)):
    os.makedirs(wdir)

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)

def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)

#make residual tower of convolutional blocks
def make_resnet(pairs=8, num_blocks=4, num_filters=32, num_outputs=1, d1=512, d2=64, word_size=word_size, ks=3, depth=5, reg_param=0.00001, final_activation='sigmoid'):

    inp = Input(shape=(int(num_blocks * word_size * 2*pairs),))
    rs = Reshape((pairs, int(2*num_blocks), word_size))(inp)
    perm = Permute((1, 3, 2))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters*2, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters*2, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2

    dense0 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return(model)


def multiple_generate_data(total_num,rounds,pairs):
    num_threads=40
    
    per_num=int(total_num/num_threads)
    arr_num=[per_num for i in range(num_threads)]
    arr_rounds=[rounds for i in range(num_threads)]
    arr_pairs=[pairs for i in range(num_threads)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(sk.make_train_data, arr_num,arr_rounds,arr_pairs))
    
    X=np.zeros((len(results[0][0])*num_threads,len(results[0][0][0])),dtype=np.uint8)
    Y=np.zeros(len(results[0][1])*num_threads,dtype=np.uint8)
    
    for i in range(num_threads):
        X[i*per_num:(i+1)*per_num]=results[i][0]
        Y[i*per_num:(i+1)*per_num]=results[i][1]
    
    del results
    gc.collect()
    return X,Y
    



def train_speck_distinguisher(num_epochs, num_rounds, depth=5, pairs=8):
    
    print("pairs = ", pairs)
    print("num_rounds = ", num_rounds)

    
    batch_size = 30000

    net = make_resnet(pairs=pairs, depth=depth, reg_param=10**-5,word_size=sk.WORD_SIZE())
        # net.summary()
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # X, Y = sk.make_train_data(2*10**7, num_rounds, pairs=pairs)
    # X_eval, Y_eval = sk.make_train_data(2*10**6, num_rounds, pairs=pairs)
    

    X,Y=multiple_generate_data(2*10**7,num_rounds,pairs)
    X_eval,Y_eval=multiple_generate_data(2*10**6,num_rounds,pairs)

  
    print("multiple processing end ......")

  
    # print("make data over")
    
    src = wdir+'simeck'+str(sk.WORD_SIZE()*2)+'_best_model_'+str(num_rounds)+'r_depth'+str(depth)+"_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)
    check = make_checkpoint(src+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size,
                validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    # net.save(wdir+'model_'+str(num_rounds)+'r_depth'+str(depth) +
    #      "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+'.h5')
         
    dump(h.history, open(wdir+'simeck'+str(sk.WORD_SIZE()*2)+'_hist'+str(num_rounds)+'r_depth'+str(depth) +
         "_num_epochs"+str(num_epochs)+"_pairs"+str(pairs)+"_acc_"+str(np.max(h.history['val_acc']))+'.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    
    # 重命名文件
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src +'.h5' , dst+'.h5')
    
    # return(net, h)


if __name__ == "__main__":
    
    # rounds=[15,16,17,18]
    rounds=[10]
    pairs = 8
    for r in rounds:
        train_speck_distinguisher(num_epochs=20, num_rounds=r, depth=5, pairs=pairs)
