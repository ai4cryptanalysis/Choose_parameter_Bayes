import os
NUM=1#
os.environ["CUDA_VISIBLE_DEVICES"] = str(NUM)#
import simeck as sk
import numpy as np

from tensorflow.keras.models import load_model
from os import urandom
import gc

import concurrent.futures
WORD_SIZE = sk.WORD_SIZE()



#nr
def generate_ciphertext(pairs,n,nr,diff=(0x0000, 0x0040)):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1)
    ks = sk.expand_key_simeck(keys, nr+1)
    ks=[np.repeat(i,pairs) for i in ks]#
  
    pt0l = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    pt0r = np.frombuffer(urandom(2*n*pairs), dtype=np.uint16)
    # print("pt0l.shape",pt0l.shape)
    pt1l = pt0l ^ diff[0]
    pt1r = pt0r ^ diff[1]
    
    # 
    ct0l, ct0r = sk.encrypt_simeck((pt0l, pt0r), ks)
    ct1l, ct1r = sk.encrypt_simeck((pt1l, pt1r), ks)
    
    return [ct0l, ct0r,ct1l, ct1r],ks[-1]


def multiple_convert_to_binary(arr):
    num_threads=16
    n=len(arr[0])
    
    per_num=int(n/num_threads)
   
    arr_multi=[[data[i*per_num:(i+1)*per_num] for data in arr]   for i in range(num_threads)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(sk.convert_to_binary, arr_multi))

    # 
    per_num=len(results[0])
    X=np.zeros((len(results[0])*num_threads,len(results[0][0])),dtype=np.uint8)

    for i in range(num_threads):
        X[i*per_num:(i+1)*per_num] = results[i]
    
    del results
    gc.collect()
    #print('Threads='+str(num_threads))

    return X


def decrypt(C,rk,pairs=8):
    ctdata0l,ctdata0r,ctdata1l,ctdata1r=C[0],C[1],C[2],C[3]
    ctdata0l,ctdata0r = sk.dec_one_round_simeck((ctdata0l, ctdata0r), rk)
    ctdata1l,ctdata1r = sk.dec_one_round_simeck((ctdata1l, ctdata1r), rk)
    
    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
 
    
    secondLast_ctdata0r = sk.rol(ctdata0r, 5) & sk.rol(ctdata0r, 0) ^ sk.rol(ctdata0r, 1) ^ ctdata0l
    secondLast_ctdata1r = sk.rol(ctdata1r, 5) & sk.rol(ctdata1r, 0) ^ sk.rol(ctdata1r, 1) ^ ctdata1l
    
    delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
    
    thirdLast_ctdata0r = ctdata0r ^ sk.rol(secondLast_ctdata0r,5) & sk.rol(secondLast_ctdata0r,0) ^ sk.rol(secondLast_ctdata0r,1)
    thirdLast_ctdata1r = ctdata1r ^ sk.rol(secondLast_ctdata1r,5) & sk.rol(secondLast_ctdata1r,0) ^ sk.rol(secondLast_ctdata1r,1)    
        
    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    X = multiple_convert_to_binary([delta_ctdata0l,delta_ctdata0r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_secondLast_ctdata0r,delta_thirdLast_ctdata0r]);
    
    del ctdata0l,ctdata0r,ctdata1l,ctdata1r
    del delta_ctdata0l,delta_ctdata0r,delta_secondLast_ctdata0r,delta_thirdLast_ctdata0r
    gc.collect()
    
    l=len(X[0])
    X=X.flatten()
    X=X.reshape(-1,pairs*l)
    return X
    
    
    

def generate_WKRP():
    per_diff_cipher_num=2**14#
    pairs=8
    nr=11#
    net=load_model("./good_trained_nets/simeck32_best_model_11r_depth5_num_epochs20_pairs8_acc_0.5656124949455261.h5")
    
    Key_diff=np.arange(2**WORD_SIZE)
    Key_diff=np.array(Key_diff,dtype=np.uint16)
    
    per_diff_num=2**12#
    
    mean=np.zeros(2**WORD_SIZE,dtype=np.float64)
    std=np.zeros(2**WORD_SIZE,dtype=np.float64)
    
    for i in range(int(len(Key_diff)/per_diff_num)):
        print(i+1,int(len(Key_diff)/per_diff_num))
        Choose_diff=np.array(Key_diff[i*per_diff_num:(i+1)*per_diff_num])
        Extend_diff=np.repeat(Choose_diff,pairs*per_diff_cipher_num)
        C,last_key=generate_ciphertext(pairs=pairs,n=per_diff_cipher_num*per_diff_num,nr=nr)
        last_key=last_key^Extend_diff
        X=decrypt(C,last_key)
        #print(hex(Choose_diff[-1]))
        #return last_key
        

        Z=net.predict(X,batch_size=2**15)
        Z = np.array(Z).flatten()
        Z=Z.reshape(per_diff_num,per_diff_cipher_num)
        
        per_mean= np.mean(Z,axis=1)
        per_std = np.std(Z,axis=1)
        
        mean[i*per_diff_num:(i+1)*per_diff_num]=per_mean.flatten()
        std[i*per_diff_num:(i+1)*per_diff_num]=per_std.flatten()
        
        del Extend_diff,C,last_key,X,Z
        gc.collect()
    wdir = "./WKRP/"
    np.save(wdir+"simeck"+str(sk.WORD_SIZE()*2)+"_data_wrong_key_mean_"+str(nr)+"r_pairs"+str(pairs)+".npy",mean)
    np.save(wdir+"simeck"+str(sk.WORD_SIZE()*2)+"_data_wrong_key_std_"+str(nr)+"r_pairs"+str(pairs)+".npy",std)
    return 0

a=generate_WKRP()        
    
    
    


