import os
NUM=0#
os.environ["CUDA_VISIBLE_DEVICES"] = str(NUM)#
import gc

from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import simeck as sk
import numpy as np

import concurrent.futures

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')#
tf.config.experimental.set_memory_growth(physical_devices[0], True)

wdir = './good_trained_nets/'
word_size = sk.WORD_SIZE()


batch_size = 30000

def cyclic_lr(num_epochs, high_lr, low_lr):
    def res(i): return low_lr + ((num_epochs-1) - i %
                                 num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return(res)
def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return(res)

def multiple_generate_data(total_num,rounds,pairs,diff):
    num_threads=40
    
    per_num=int(total_num/num_threads)
    arr_num=[per_num for i in range(num_threads)]
    arr_rounds=[rounds for i in range(num_threads)]
    arr_pairs=[pairs for i in range(num_threads)]
    arr_diff=[diff for i in range(num_threads)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(sk.make_train_data, arr_num,arr_rounds,arr_pairs,arr_diff))
    
    X=np.zeros((len(results[0][0])*num_threads,len(results[0][0][0])),dtype=np.uint8)
    Y=np.zeros(len(results[0][1])*num_threads,dtype=np.uint8)
    
    for i in range(num_threads):
        X[i*per_num:(i+1)*per_num]=results[i][0]
        Y[i*per_num:(i+1)*per_num]=results[i][1]
    
    del results
    gc.collect()
    return X,Y

def first_stage(total_num,num_rounds=12,pairs=8):

 
    X, Y = multiple_generate_data(total_num, rounds=num_rounds-3, pairs=pairs,diff=(0x0140, 0x0080))
    X_eval,Y_eval= multiple_generate_data(int(total_num/10), rounds=num_rounds-3, pairs=pairs,diff=(0x0140, 0x0080))
    
    
    net = load_model(wdir+"simeck32_best_model_11r_depth5_num_epochs20_pairs8_acc_0.5656124949455261.h5")
    net_json = net.to_json()
    net_first = model_from_json(net_json)
    net_first.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net_first.load_weights(wdir+"simeck32_best_model_11r_depth5_num_epochs20_pairs8_acc_0.5656124949455261.h5")

    check = make_checkpoint(
        wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    net_first.fit(X, Y, epochs=10, batch_size=batch_size,
                  validation_data=(X_eval, Y_eval),callbacks=[lr,check])

    print("################################################")
    # net_first.save(wdir+"net_first.h5")


def second_stage(total_num,num_rounds=12, pairs=8):


    X, Y = multiple_generate_data(total_num, rounds=num_rounds, pairs=pairs,diff=(0x0000,0x0040))
    X_eval, Y_eval = multiple_generate_data(int(total_num/10), rounds=num_rounds, pairs=pairs,diff=(0x0000,0x0040))
    
    net = load_model(wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
    net_json = net.to_json()

    net_second = model_from_json(net_json)
    # net_second.compile(optimizer=Adam(learning_rate = 10**-4), loss='mse', metrics=['acc'])
    net_second.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net_second.load_weights(wdir+'first_best'+str(num_rounds)+"_pairs"+str(pairs)+'.h5')
        
    
    check = make_checkpoint(
        wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.0001, 0.00001))
    net_second.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[lr,check])
    print("################################################")
    # net_second.save(wdir+"net_second.h5")


def stage_train(total_num,num_rounds=12, pairs=8):


    X, Y = multiple_generate_data(total_num, rounds=num_rounds, pairs=pairs,diff=(0x0000,0x0040))
    X_eval, Y_eval = multiple_generate_data(int(total_num/10), rounds=num_rounds, pairs=pairs,diff=(0x0000,0x0040))
    
    net = load_model( wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    net_json = net.to_json()

    net_third = model_from_json(net_json)
    net_third.compile(optimizer=Adam(learning_rate = 10**-5), loss='mse', metrics=['acc'])
    net_third.load_weights( wdir+'second_best'+str(num_rounds)+"r_pairs"+str(pairs)+'.h5')
    
    src = "staged_simeck"+str(sk.WORD_SIZE()*2)+"_best_model_"+str(num_rounds)+"r_depth5_num_epochs10_pairs"+str(pairs)
    check = make_checkpoint(wdir+src+'.h5')
    h = net_third.fit(X, Y, epochs=10, batch_size=batch_size,
                   validation_data=(X_eval, Y_eval),callbacks=[check])
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(wdir+src +'.h5' , wdir+dst+'.h5')
    print("Best validation accuracy: ", np.max(h.history['val_acc']))

    # net_third.save(wdir+"simon_model_"+str(num_rounds)+"r_depth5_num_epochs20_pairs"+str(pairs)+".h5")
   

for i in range(1):

    first_stage(total_num=2*10**7, num_rounds=12,pairs=8)
    second_stage(total_num=2*10**7,num_rounds=12,pairs=8)
    stage_train( total_num=2*10**7,num_rounds=12,pairs=8)