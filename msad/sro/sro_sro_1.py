import numpy as np
import math
from random import randrange 
from itertools import combinations
import time
import multiprocessing as mp

# !Cr, Co, Ni {0, 1, -1}
cr_, co_, ni_ = 1/3, 1/3, 1/3
dir = ''
ind_1nn = np.load('fcc_108/ind_1nn.npy')
ind_2nn = np.load('fcc_108/ind_2nn.npy')
ind_3nn = np.load('fcc_108/ind_3nn.npy')
ind_4nn = np.load('fcc_108/ind_4nn.npy')
ind_raw = np.load('fcc_108/ind_raw.npy')
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

def phi1(x):
    return math.sqrt(3/2)*x

def phi2(x):
    return math.sqrt(2)*(3/2*(x**2) - 1)

def cpr(val1, val2):

    basis_11 = phi1(val1)*phi1(val2)
    basis_12 = (phi1(val1)*phi2(val2)+phi2(val1)*phi1(val2))/2
    basis_22 = phi2(val1)*phi2(val2)

    return np.array([basis_11, basis_12, basis_22])

def cor_func(ind_nNN, ele_list):
    cor_func_1, cor_func_2, cor_func_3 = 0, 0, 0
    for i in ind_nNN:
        cor_func = cpr(ele_list[i[0]], ele_list[i[1]])
        cor_func_1 += cor_func[0]
        cor_func_2 += cor_func[1]
        cor_func_3 += cor_func[2]
    return np.array([cor_func_1, cor_func_2, cor_func_3])

def ele_list_gen(cr_content, co_content, ni_content, mode='int'):
    assert cr_content+co_content+ni_content==1, 'Make sure atomic ratio sum to 1'

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

def ideal_cor_func(cr_content, co_content, bond_num, mode = 'dont print bro'):
    ni_content = 1-cr_content-co_content
    
    num_crcr = cr_content**2*bond_num
    num_coco = co_content**2*bond_num
    num_nini = ni_content**2*bond_num
    num_crco = 2*cr_content*co_content*bond_num
    num_coni = 2*co_content*ni_content*bond_num
    num_crni = 2*cr_content*ni_content*bond_num
    
    num_true = num_crcr+num_coco+num_nini+num_crco+num_coni+num_crni
    assert int(num_true)==bond_num, f'Current num: {num_true}, Make sure bond sum to ture value'
    cor_func = (num_crcr*cpr(0,0)
               +num_coco*cpr(1,1)
               +num_nini*cpr(-1,-1)
               +num_crco*cpr(0,1)
               +num_crni*cpr(0,-1)
               +num_coni*cpr(1,-1))
    if mode == 'please print man!':
        print(f'ideal cor func of Cr{cr_content*100}Co{co_content*100}Ni{ni_content*100}: {cor_func}')
    return cor_func

def sro_paramfind(ind_nNN, state, cr_, co_, ni_):
    num_s = len(ind_nNN)
    n_nini, n_nicr, n_nico, n_crcr, n_coco, n_crco = 0,0,0,0,0,0
    for i in ind_nNN:
        pair_list = np.array([state[i[0]], state[i[1]]])
        if np.linalg.norm(pair_list-nini) == 0:
            n_nini += 1
        elif np.linalg.norm(pair_list-nicr) == 0:
            n_nicr += 1
        elif np.linalg.norm(pair_list-nico) == 0:
            n_nico += 1
        elif np.linalg.norm(pair_list-crcr) == 0:
            n_crcr += 1
        elif np.linalg.norm(pair_list-crco) == 0:
            n_crco += 1
        elif np.linalg.norm(pair_list-coco) == 0:
            n_coco += 1

    a_nini = 1 - n_nini/num_s/(ni_*ni_)
    a_nicr = 1 - n_nicr/num_s/(ni_*cr_)
    a_nico = 1 - n_nico/num_s/(ni_*co_)
    a_crcr = 1 - n_crcr/num_s/(cr_*cr_)
    a_crco = 1 - n_crco/num_s/(cr_*co_)
    a_coco = 1 - n_coco/num_s/(co_*co_)

    return np.array([a_nini, a_nicr, a_nico, a_crcr, a_crco, a_coco])

# !ideal sro list
ideal_sro = np.array([
    -0.002, -0.073, 0.075, 0.182, -0.108, 0.033
])

def single_test(iter):
    np.random.seed()
    ele_list = ele_list_gen(1/3, 1/3, 1/3, mode='int')
    sro_param = sro_paramfind(ind_1nn, ele_list, 1/3, 1/3, 1/3)

    if iter % 10000 == 0:
        print(iter)
    if np.linalg.norm(sro_param - ideal_sro) <= 0.04:
        return ele_list.tolist()
    else:
        pass

def multicore(iter_time, process_num):
    # pool = mp.Pool(processes=2)#自动分配进程/核
    pool = mp.Pool(processes=process_num)
    output_list = [pool.map(single_test, range(iter_time))]
    # map equation to the value
    return output_list

if __name__ == '__main__':
    iter_time = 10000000
    start_ = time.time()
    output_list = [multicore(iter_time, process_num=6)][0][0]
    output_list = [i for i in output_list if i]
    np.save(f'sro_list3333_1.npy', output_list)
    second_to_hour(time.time() - start_)
    try:
        print(len(output_list))
    except:
        print('sadly nothing reached goal')




