#Proof of concept implementation of 16-round key recovery attack of simeck32/64

import simeck as cipher
from simeck import rol
import numpy as np
from datetime import datetime
import psutil #CPU INFO
import cpuinfo #CPU INFO
import platform #CPU INFO
import gc
from os import urandom
from math import log2,sqrt
from time import time

import os
number=input("Choose GPU:")
NUM=int(number)
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)#

import tensorflow as tf
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')#
tf.config.experimental.set_memory_growth(physical_devices[0], True)

WORD_SIZE = cipher.WORD_SIZE()

a_circle = 5
b_circle = 0
c_circle = 1

#binarize a given ciphertext sample
#ciphertext is given as a sequence of arrays
#each array entry contains one word of ciphertext for all ciphertexts given
def convert_to_binary(l):
    n = len(l);
    k = WORD_SIZE * n
    X = np.zeros((k, len(l[0])),dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE
        offset = WORD_SIZE - 1 - i%WORD_SIZE
        X[i] = (l[index] >> offset) & 1
    X = X.transpose();
    return(X)

def hw(v):
    res = np.zeros(v.shape,dtype=np.uint8)
    for i in range(16):
        res = res + ((v >> i) & 1)
    return(res);

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16)
low_weight = low_weight[hw(low_weight) <= 2]

#make a plaintext structure
#takes as input a sequence of plaintexts, a desired plaintext input difference, and a set of neutral bits

def obtain_sys_info():
    # CPU INFO
    print("="*10, 'SYSTEM INFO',"="*10, file=logfile, flush=True)
    info = cpuinfo.get_cpu_info()
    CPU_ver=info["brand_raw"]
    print(CPU_ver, file=logfile, flush=True)
    print("Physical cores:", psutil.cpu_count(logical=False), file=logfile, flush=True)
    print("Total cores:", psutil.cpu_count(logical=True), file=logfile, flush=True)
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz", file=logfile, flush=True)
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz", file=logfile, flush=True)

    uname = platform.uname()
    print(f"System: {uname.system}", file=logfile, flush=True)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"GPU ：{details['device_name']}", file=logfile, flush=True)
        

    #print(divices, file=logfile, flush=True)    
    print("="*30, file=logfile, flush=True)

def generate_nb_diff(nb):
    index=0
    for i in nb:
        index=index^(1<<i)
    return index

def make_structure(pt0, pt1, diff,neutral_bits):
    
    p0 = np.copy(pt0); p1 = np.copy(pt1);
    p0 = p0.reshape(-1,1); p1 = p1.reshape(-1,1)
    for i in neutral_bits:
        d = generate_nb_diff(i); 
        d0 = d >> 16; 
        d1 = d & 0xffff
        p0 = np.concatenate([p0,p0^d0],axis=1);
        p1 = np.concatenate([p1,p1^d1],axis=1);
    p0b = p0 ^ diff[0]; p1b = p1 ^ diff[1];
    return(p0,p1,p0b,p1b);

#generate a Speck key, return expanded key
def gen_key(nr):
    key = np.frombuffer(urandom(8),dtype=np.uint16).reshape(4,-1)
    ks = cipher.expand_key_simeck(key, nr)
    return(ks)

def gen_plain(n):
    pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
    return(pt0, pt1);

def find_good(cts, key, nr=1+3, target_diff = (0x0,0x40)):
    pt0a, pt1a = cipher.decrypt_simeck((cts[0], cts[1]), key[nr:]);
    pt0b, pt1b = cipher.decrypt_simeck((cts[2], cts[3]), key[nr:]);
    diff0 = pt0a ^ pt0b; diff1 = pt1a ^ pt1b;
    d0 = (diff0 == target_diff[0]); d1 = (diff1 == target_diff[1]);
    d = d0 * d1;
    v = np.sum(d,axis=1);
    return(v)


def gen_challenge(load_key,existing_key):
    
    n=2**9
    
    #print('Using '+str(n)+' Ciphertext Structures',file=logfile, flush=True)
    nr=1+3+11+1
    
    diff=(0x140, 0x200)
    neutral_bits = [[3],[4],[5],
                    [7],[8],[9],[13],[14],
                    [15],[18],[20],[22],[24]]
    
    if(load_key==True):
        key = existing_key
    else:
        key = gen_key(nr)
    
    '''
    flag=True
    while(flag):
        pt0, pt1 = gen_plain(n);
        pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
        pt0a, pt1a = cipher.dec_one_round_simeck((pt0a, pt1a),0)
        pt0b, pt1b = cipher.dec_one_round_simeck((pt0b, pt1b),0)
        
  
        ct0a, ct1a = cipher.encrypt_simeck((pt0a, pt1a), key)
        ct0b, ct1b = cipher.encrypt_simeck((pt0b, pt1b), key)
        
        v=find_good([ct0a, ct1a, ct0b, ct1b], key)
        if(max(v)==2**len(neutral_bits)):
            flag=False
    '''
    pt0, pt1 = gen_plain(n);
    pt0a, pt1a, pt0b, pt1b = make_structure(pt0, pt1, diff=diff, neutral_bits=neutral_bits)
    pt0a, pt1a = cipher.dec_one_round_simeck((pt0a, pt1a),0)
    pt0b, pt1b = cipher.dec_one_round_simeck((pt0b, pt1b),0)
        
  
    ct0a, ct1a = cipher.encrypt_simeck((pt0a, pt1a), key)
    ct0b, ct1b = cipher.encrypt_simeck((pt0b, pt1b), key)
        
    v=find_good([ct0a, ct1a, ct0b, ct1b], key)

    '''
    v=find_good([ct0a, ct1a, ct0b, ct1b], key)
    index=np.where(v==2**len(neutral_bits))[0]
    for i in range(len(index)):
        ct0a[i]=ct0a[index[i]]
        ct1a[i]=ct1a[index[i]]
        ct0b[i]=ct0b[index[i]]
        ct1b[i]=ct1b[index[i]]
    
    v=find_good([ct0a, ct1a, ct0b, ct1b], key)
    index=np.where(v==2**len(neutral_bits))[0]
    for i in range(len(index)):
        ct0a[i]=ct0a[index[i]]
        ct1a[i]=ct1a[index[i]]
        ct0b[i]=ct0b[index[i]]
        ct1b[i]=ct1b[index[i]] 
    
    v=find_good([ct0a, ct1a, ct0b, ct1b], key)
    index=np.where(v==2**len(neutral_bits))[0]
    for i in range(len(index)):
        ct0a[i]=ct0a[index[i]]
        ct1a[i]=ct1a[index[i]]
        ct0b[i]=ct0b[index[i]]
        ct1b[i]=ct1b[index[i]]
        
    v=find_good([ct0a, ct1a, ct0b, ct1b], key)
    '''
    return([ct0a, ct1a, ct0b, ct1b], key,v)



#cts,key,v=gen_challenge()

'''
for i in range(2**9):
    cts,key=gen_challenge()
    v=find_good(cts, key)
    print(max(v),end=', ')

'''

def cipher2multi(C,pair=8):
    ctdata0l, ctdata0r = C[0],C[1]
    ctdata1l, ctdata1r = C[2],C[3]
    
    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
 
    secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l
    secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l
    
    delta_secondLast_ctdata0r =  secondLast_ctdata0r ^ secondLast_ctdata1r
    
    thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r,a_circle) & rol(secondLast_ctdata0r,b_circle) ^ rol(secondLast_ctdata0r,c_circle)
    thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r,a_circle) & rol(secondLast_ctdata1r,b_circle) ^ rol(secondLast_ctdata1r,c_circle)    
        
    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    
    X = convert_to_binary([delta_ctdata0l,delta_ctdata0r,ctdata0l,ctdata0r,ctdata1l,ctdata1r,delta_secondLast_ctdata0r,delta_thirdLast_ctdata0r]);
    
    l=len(X[0])
    X=X.flatten()
    X=X.reshape(-1,l*pair)
    return X
    

#having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
def verifier_search(cts, best_guess,cipher_num, net):
    #print(best_guess);
    ck1 = best_guess[0] ^ low_weight;
    ck2 = best_guess[1] ^ low_weight;
    n = len(ck1);
    #cipher_num=len(cts[0])
    ck1 = np.repeat(ck1, n); keys1 = np.copy(ck1);
    ck2 = np.tile(ck2, n); keys2 = np.copy(ck2);
    ck1 = np.repeat(ck1, cipher_num);
    ck2 = np.repeat(ck2, cipher_num);
    ct0a = np.tile(cts[0], n*n);
    ct1a = np.tile(cts[1], n*n);
    ct0b = np.tile(cts[2], n*n);
    ct1b = np.tile(cts[3], n*n);
    pt0a, pt1a = cipher.dec_one_round_simeck((ct0a, ct1a), ck1);
    pt0b, pt1b = cipher.dec_one_round_simeck((ct0b, ct1b), ck1);
    pt0a, pt1a = cipher.dec_one_round_simeck((pt0a, pt1a), ck2);
    pt0b, pt1b = cipher.dec_one_round_simeck((pt0b, pt1b), ck2);
    
    
    X=cipher2multi([pt0a, pt1a, pt0b, pt1b])
    
    Z = net.predict(X, batch_size=2**14)
    Z = Z / (1 - Z);
    Z = np.log2(Z);
    Z = Z.reshape(n*n, -1);
    v = np.sum(Z, axis=1) 
    m = np.argmax(v); val = v[m];
    key1 = keys1[m]; key2 = keys2[m];
    return(key1, key2, val);



#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here


def bayesian_rank_kr(tmp_br,cand, emp_mean, m, s):
    n = len(cand)
    tmp = tmp_br ^ cand
    v = (emp_mean - m[tmp]) * s[tmp]
    v = v.reshape(-1, n)
    scores = np.linalg.norm(v, axis=1)
    return(scores)



def bayesian_key_recovery(cts, num_cand,num_iter,net, pre_mean, pre_std,tmp_br):

    #tmp_br = np.arange(2**14, dtype=np.uint16)
    #tmp_br = np.repeat(tmp_br, num_cand).reshape(-1,num_cand)
    
    #num_cand = 32#
    #num_iter=10#
    num_cipher=len(cts[0])
    keys = np.random.choice(2**(WORD_SIZE),num_cand,replace=False)& 0xcfff
    
    c0l, c0r = np.tile(cts[0], num_cand),np.tile(cts[1], num_cand)
    c1l, c1r = np.tile(cts[2], num_cand),np.tile(cts[3], num_cand)


    scores = np.zeros(2**(WORD_SIZE-2))
    all_keys = np.zeros(num_cand * num_iter,dtype=np.uint16)
    all_v = np.zeros(num_cand * num_iter)
    for i in range(num_iter):
        k = np.repeat(keys, num_cipher)
        ctdata0l, ctdata0r = cipher.dec_one_round_simeck((c0l, c0r), k)
        ctdata1l, ctdata1r = cipher.dec_one_round_simeck((c1l, c1r), k)

        X=cipher2multi([ctdata0l, ctdata0r,ctdata1l, ctdata1r])
        Z = net.predict(X,batch_size=2**14)
        del X
        gc.collect()
        
        Z = Z.reshape(num_cand, -1)
        means = np.mean(Z, axis=1)
        Z = Z/(1-Z)
        Z = np.log2(Z)
        v =np.sum(Z, axis=1) 
        #print('Checking'+str(i),max(v),end='##')
        all_v[i * num_cand:(i+1)*num_cand] = v
        all_keys[i * num_cand:(i+1)*num_cand] = np.copy(keys)
        scores = bayesian_rank_kr(tmp_br,keys, means, pre_mean, pre_std)
        tmp = np.argpartition(scores, num_cand)
        keys = tmp[0:num_cand]
        r = np.random.randint(0,4,num_cand,dtype=np.uint16)
        r = r << 12
        keys = keys ^ r
    #print('')
    return(all_keys, all_v)



def test_bayes(cts, real_key,net, net_help, 
               m_main,s_main, m_help,  s_help,
               num_cand4last,  num_inter4last,  cutoff1,
               num_cand4second,num_inter4second,cutoff2,
               max_num_key,rk1,rk2,
               use_VS=False):
    last_key=real_key[0]
    second_key=real_key[1]
    '''
    cutoff1=6.0815019607543945
    cutoff2=-21
    
    num_cand4last=32
    num_inter4last=5
    num_cand4second=32
    num_inter4second=5
    
    max_num_key=33
    '''

    print('Cand1='+str(num_cand4last),end=' ')
    print('Step1='+str(num_inter4last),end=' ')
    print('Cutoff1='+str(cutoff1),end=' --- ')
    print('Cand2='+str(num_cand4second),end=' ')
    print('Step2='+str(num_inter4second),end=' ')
    print('Cutoff2='+str(cutoff2))
    
    tmp_br_last = np.arange(2**16, dtype=np.uint16)& 0xcfff
    tmp_br_last = np.repeat(tmp_br_last, num_cand4last).reshape(-1,num_cand4last)
    
    tmp_br_second = np.arange(2**16, dtype=np.uint16)& 0xcfff
    tmp_br_second = np.repeat(tmp_br_second, num_cand4second).reshape(-1,num_cand4second)
    
    
    
    n = len(cts[0]);
    use_n=len(cts[0][0])
  
    alpha = sqrt(n);

    best_val = -10000.0; 
    best_key = (0,0); 
    best_pod = 0; 

    keys = np.random.choice(2**WORD_SIZE, 32, replace=False);
    eps = 0.00001; 
    local_best = np.full(n,-1000); 
    num_visits = np.full(n,eps);
    
    it=2**9+2**8
    
    for j in range(it):
        priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits) 
        i = np.argmax(priority)
        num_visits[i] = num_visits[i] + 1
        if (best_val > cutoff2):
            print('Last  Choose'+str(best_pod).rjust(4, '0')+' #  '+format(best_val, '.4f'),file=logfile, flush=True) 
            
            if (use_VS==False):
                return(best_key, j,best_pod,it)
 
            improvement = True
            print('')
            while improvement:
                print('\r' +'val='+str(best_val)+' # '+hex(rk1^best_key[0])+' # '+hex(rk2^best_key[1]),end='', flush=True)
                k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key,use_n ,net_help );
                improvement = (val > best_val);
                
                if (improvement):
                    best_key = (k1, k2); 
                    best_val = val;
            print('\r' +'val='+str(best_val)+' # '+hex(rk1^best_key[0])+' # '+hex(rk2^best_key[1]),end='', flush=True)
            print('')
            return(best_key, j,best_pod,it);

        
        keys, v = bayesian_key_recovery([cts[0][i], cts[1][i], cts[2][i], cts[3][i]], num_cand=num_cand4last, num_iter=num_inter4last,net=net, pre_mean=m_main, pre_std=s_main,tmp_br=tmp_br_last);
        vtmp = np.max(v);
        
        print('\r' +'Loop='+str(j)+'---Choosing '+str(i)+' Cipher',end='', flush=True)
        print(str(j)+' #  Choose'+str(i).rjust(4, '0')+' #  '+format(np.max(v), '.4f'),file=logfile, flush=True)
        
        if (vtmp > local_best[i]): 
            local_best[i] = vtmp

        if (vtmp > cutoff1):
            
            new_index=np.argsort(v*-1.0)
            v=v[new_index]
            keys=keys[new_index]
            
            l2 = [i for i in range(len(keys)) if v[i] > cutoff1]
            if(len(l2)>max_num_key):
                l2=np.argsort(np.copy(v))[-max_num_key:]
            
            print('  ---# Number of over c1 : '+str(len(l2)),file=logfile, flush=True)
            for i2 in l2:
                c0a, c1a = cipher.dec_one_round_simeck((cts[0][i][:2048],cts[1][i][:2048]),keys[i2])
                c0b, c1b = cipher.dec_one_round_simeck((cts[2][i][:2048],cts[3][i][:2048]),keys[i2])         
                keys2,v2 = bayesian_key_recovery([c0a, c1a, c0b, c1b],num_cand=num_cand4second, num_iter=num_inter4second,net=net_help, pre_mean=m_help, pre_std=s_help,tmp_br=tmp_br_second)
                vtmp2 = np.max(v2);
                
                key_candidates='0x'+hex(keys[i2]^last_key)[2:].rjust(4, '0')
                key_candidates=key_candidates+', 0x'+hex(keys2[np.argmax(v2)]^second_key)[2:].rjust(4, '0')
                
                print('  ---# '+'BKS1: '+format(v[i2], '.4f')+'  '+'BKS2: '+format(np.max(vtmp2), '.4f')+'  '+key_candidates,file=logfile, flush=True)
                if (vtmp2 > best_val):
                    best_val = vtmp2; 
                    best_key = (keys[i2], keys2[np.argmax(v2)]); 
                    best_pod=i;
    
    print('Last  Choose'+str(best_pod).rjust(4, '0')+' #  '+format(best_val, '.4f'),file=logfile, flush=True) 
    if (use_VS==False):
        return(best_key, it,best_pod,it)
   
    improvement = True
    while improvement:
        k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key,use_n ,net_help);
        improvement = (val > best_val);
        if (improvement):
            best_key = (k1, k2); 
            best_val = val;
    print('\r' +'val='+str(best_val)+' # '+hex(rk1^best_key[0])+' # '+hex(rk2^best_key[1]),end='', flush=True)
    print('')
    #return(it,best_pod)
    return(best_key, it,best_pod,it)
   

def make_fileFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created")
    else:
        print("Folder already exists")

def print_attack_par(par_attack):
    num_cand1,step1,cutoff1=par_attack[0],par_attack[1],par_attack[2]
    num_cand2,step2,cutoff2=par_attack[3],par_attack[4],par_attack[5]

    print('Cand1='+str(num_cand1),', Step1='+str(step1),', c1='+str(cutoff1),end='  ###  ')
    print('Cand2='+str(num_cand2),', Step2='+str(step2),', c2='+str(cutoff2))
    
    print('Cand1='+str(num_cand1),', Step1='+str(step1),', c1='+str(cutoff1),file=logfile, flush=True)
    print('Cand2='+str(num_cand2),', Step2='+str(step2),', c2='+str(cutoff2),file=logfile, flush=True)

def test(n):
    
    #load distinguishers
    wdir = './good_trained_nets/'
    net_name='simeck32_best_model_11r_depth5_num_epochs20_pairs8_acc_0.5656124949455261.h5'
    net = tf.keras.models.load_model( wdir+net_name)
        
    net_name='simeck32_best_model_10r_depth5_num_epochs20_pairs8_acc_0.7203624844551086.h5'
    net_help = tf.keras.models.load_model( wdir+net_name)
        
    path='./WKRP/'
    m_main=np.load(path+"simeck32_data_wrong_key_mean_11r_pairs8.npy")
    s_main=np.load(path+"simeck32_data_wrong_key_std_11r_pairs8.npy")
    s_main=1.0/s_main
        
    m_help=np.load(path+"simeck32_data_wrong_key_mean_10r_pairs8.npy")
    s_help=np.load(path+"simeck32_data_wrong_key_std_10r_pairs8.npy")
    s_help=1.0/s_help
    
    
    
    cutoff1 = 0.5656
    cutoff2 = -135
    
    num_cand4last=16
    num_inter4last=5
    num_cand4second=16
    num_inter4second=5
    
    par=[num_cand4last,  num_inter4last,  cutoff1,
         num_cand4second,num_inter4second,cutoff2]
    
    max_num_key=num_cand4last*num_inter4last
    
    key_attack_path='./Attack_result_'+str(num_cand4last)+'_'+str(num_inter4last)+'c1='+str(cutoff1)+'_'+str(num_cand4second)+'_'+str(num_inter4second)+'_'+'c2='+str(cutoff2)+'_'+str(max_num_key)
    make_fileFolder(key_attack_path)
    key_attack_path=key_attack_path+'/'
    
    
    
    now = datetime.now()
    current_time = now.strftime("%Y:%m:%d:%H:%M:%S:")
    current_time = current_time.replace(":", "_")
    
    file_name='GPU'+str(NUM)+'Cand1='+str(num_cand4last)+'Step1='+str(num_inter4last)
    file_name=file_name+'Cand2='+str(num_cand4second)+'Step2='+str(num_inter4second)+'_'+current_time
    
    global logfile
    logfile = open(key_attack_path+file_name+ '.txt', 'w+')
    
    
    #arr1 = np.zeros(n, dtype=np.uint16); arr2 = np.zeros(n, dtype=np.uint16);
    print_attack_par(par)
    obtain_sys_info()
    
    choose_plain=[]


    
    t0 = time()
    for i in range(n):
        print("Test:",i);
        
        print(" ",file=logfile, flush=True)
        print("=*"*15,file=logfile, flush=True)
        print_attack_par(par)
        print("Test:",i,file=logfile, flush=True)
        ct, key,v = gen_challenge(False,None)
        
        #v=find_good(ct, key)

        print('Generating Ciphertext Data: ',np.where(v==max(v))[0])        
        print('Right Structure',np.where(v==max(v))[0],file=logfile, flush=True)
        #test_bayes(cts, net, net_help, m_main,s_main, m_help,  s_help,num_cand4last,num_inter4last,cutoff1,num_cand4second,num_inter4second,cutoff2,max_num_key)
        guess_key,num_used,best_index,max_it = test_bayes(ct,  real_key=(key[-1][0],key[-2][0]),
                                         net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help,
                                         num_cand4last=num_cand4last,    num_inter4last=num_inter4last,    cutoff1=cutoff1,
                                         num_cand4second=num_cand4second,num_inter4second=num_inter4second,cutoff2=cutoff2,
                                         max_num_key=max_num_key,rk1=key[-1][0],rk2=key[-2][0])

        
        
        
        if(num_used==max_it):
            ct, key,v = gen_challenge(True,key)
            
            #v=find_good(ct, key)
            print('')
            print('Re_Generating Ciphertext Data: ',np.where(v==max(v))[0])
            print('   =====',file=logfile, flush=True)
            print('Re_Right Structure',np.where(v==max(v))[0],file=logfile, flush=True)
            #test_bayes(cts, net, net_help, m_main,s_main, m_help,  s_help,num_cand4last,num_inter4last,cutoff1,num_cand4second,num_inter4second,cutoff2,max_num_key)
            guess_key,num_used,best_index,max_it = test_bayes(ct,  real_key=(key[-1][0],key[-2][0]),
                                             net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help,
                                             num_cand4last=num_cand4last,    num_inter4last=num_inter4last,    cutoff1=cutoff1,
                                             num_cand4second=num_cand4second,num_inter4second=num_inter4second,cutoff2=cutoff2,
                                             max_num_key=max_num_key,rk1=key[-1][0],rk2=key[-2][0])
            
        
        
        #arr1[i] = guess[0] ^ key[-1][0]; arr2[i] = guess[1] ^ key[-2][0];
        #print("Difference between real key and key guess: ", hex(arr1[i]), hex(arr2[i]));
        print('Choose the plaintext structure',best_index,v[best_index],max(v))
        print('Choose the plaintext structure',best_index,v[best_index],max(v),file=logfile, flush=True)
        choose_plain.append([v[best_index],max(v)])
        t1=time()
        #print("Wall time per attack (average in seconds):", (t1 - t0)/(i+1))
        #print("Wall time per attack (average in seconds):", (t1 - t0)/(i+1),file=logfile, flush=True)
        print('---------------------------------------')
    
    t1 = time()
    print("Done.")
    #d1 = [hex(x) for x in arr1]; d2 = [hex(x) for x in arr2]
    #print("Differences between guessed and last key:", d1)
    #print("Differences between guessed and second-to-last key:", d2)
    #print("Wall time per attack (average in seconds):", (t1 - t0)/n)

    '''
    pre=str(NUM)
    np.save(key_attack_path+'Using_arr1_'+pre+'.npy',arr1)
    np.save(key_attack_path+'Using_arr2_'+pre+'.npy',arr2)
    np.save(key_attack_path+'Using_choose_plain_'+pre+'.npy',np.array(choose_plain))
    np.save(key_attack_path+'Using_Time_'+pre+'.npy',np.array([(t1 - t0)/n]))
    '''
    
    return 0

rrr=test(15)
