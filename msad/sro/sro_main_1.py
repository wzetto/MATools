import numpy as np
import math
from random import randrange 
from itertools import combinations
import time
import multiprocessing as mp
from fitter import Fitter
import matplotlib.pyplot as plt

# !Cr, Co, Ni {0, 1, -1}
cr_, co_, ni_ = 1/3, 1/3, 1/3
dir = ''
ind_1nn = np.load('./ind_1nn.npy')
nini = np.array([-1,-1])
nicr = np.array([-1,0])
nico = np.array([-1,1])
crcr = np.array([0,0])
crco = np.array([0,1])
coco = np.array([1,1])
# ideal_1, ideal_2, ideal_3, ideal_4 = 0,0,0,0

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

def ele_list_gen(cr_content, co_content, ni_content, mode='int'):
    assert cr_content+co_content+ni_content==1, 'Make sure atomic ratio summing to 1'

    if mode == 'randchoice':
        len_cr = randrange(int(cr_content*108),int(cr_content*108)+2)
        len_co = randrange(int(co_content*108),int(co_content*108)+2)
    elif mode == 'int':
        len_cr = int(cr_content*108)
        len_co = int(co_content*108) 
    
    len_ni = 108-len_cr-len_co
    if abs(len_ni-108*ni_content) <= 1:
        ele_list_raw = np.concatenate([np.zeros(len_cr),np.ones(len_co),0-np.ones(len_ni)],axis=0)
        np.random.shuffle(ele_list_raw)
        return ele_list_raw

def sro_paramfind(ind_nNN, state, cr_, co_, ni_):
    #* alpha = 1 - P_AB/cB, finding B given A
    num_s = len(ind_nNN)
    num_cr, num_co, num_ni = 0, 0, 0
    n_nini, n_nicr, n_crni, n_nico, n_coni, n_crcr, n_coco, n_crco, n_cocr = 0,0,0,0,0,0,0,0,0
    for i in ind_nNN:
        pair_list = np.sort(np.array([state[i[0]], state[i[1]]]))
        
        if np.linalg.norm(pair_list-nini) == 0:
            num_ni += 2
            n_nini += 2

        elif np.linalg.norm(pair_list-nicr) == 0:
            num_ni += 1
            num_cr += 1
            n_nicr += 1
            n_crni += 1

        elif np.linalg.norm(pair_list-nico) == 0:
            num_ni += 1
            num_co += 1
            n_nico += 1
            n_coni += 1

        elif np.linalg.norm(pair_list-crcr) == 0:
            num_cr += 2
            n_crcr += 2

        elif np.linalg.norm(pair_list-crco) == 0:
            num_cr += 1
            num_co += 1
            n_crco += 1
            n_cocr += 1

        elif np.linalg.norm(pair_list-coco) == 0:
            num_co += 2
            n_coco += 2

    a_nini = 1 - n_nini/num_ni/(ni_)
    a_crni = 1 - n_crni/num_cr/(ni_)
    a_nicr = 1 - n_nicr/num_ni/(cr_)
    a_nico = 1 - n_nico/num_ni/(co_)
    a_coni = 1 - n_coni/num_co/(ni_)
    a_crcr = 1 - n_crcr/num_cr/(cr_)
    a_crco = 1 - n_crco/num_cr/(co_)
    a_cocr = 1 - n_cocr/num_co/(cr_)
    a_coco = 1 - n_coco/num_co/(co_)

    return np.array([a_crcr, a_crni, a_crco, a_nicr, a_cocr, a_nini, a_nico, a_coco, a_coni])

# !ideal sro list
# ideal_sro = np.array([
#     -0.002, -0.073, 0.075, 0.182, -0.108, 0.033
# ])

def single_test(iter):
    np.random.seed()
    ele_list = ele_list_gen(1/3, 1/3, 1/3, mode='int')
    sro_param = sro_paramfind(ind_1nn, ele_list, 1/3, 1/3, 1/3)

    if iter % 50000 == 0:
        print(iter)
    return sro_param.tolist(), ele_list.tolist()

def multicore(iter_time, process_num):
    # pool = mp.Pool(processes=2)
    pool = mp.Pool(processes=process_num)
    output_list = [pool.map(single_test, range(iter_time))]
    # map equation to the value
    return output_list

if __name__ == '__main__':
    iter_time = 5000000
    start_ = time.time()
    output_list = [multicore(iter_time, process_num=20)][0][0]
    output_list = [i for i in output_list if i]
    np.save(f'./rawlist_cr{int(cr_*100)}co{int(co_*100)}_6.npy', output_list)
    second_to_hour(time.time() - start_)
    try:
        print(len(output_list))
    except:
        print('Sadly nothing reached the goal')
