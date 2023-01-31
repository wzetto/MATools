import math
import numpy as np
from random import randrange

def find_overlap(A, B):

    if not A.dtype == B.dtype:
        raise TypeError("A and B must have the same dtype")
    if not A.shape[1:] == B.shape[1:]:
        raise ValueError("the shapes of A and B must be identical apart from "
                         "the row dimension")

    A = np.ascontiguousarray(A.reshape(A.shape[0], -1))
    B = np.ascontiguousarray(B.reshape(B.shape[0], -1))

    t = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))

    return np.in1d(A.view(t), B.view(t))

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

def phi1(x):
    return 2/math.sqrt(10)*x

def phi2(x):
    return -5/3 + 2/3*(x**2)

def phi3(x):
    return -17/30*math.sqrt(10)*x + math.sqrt(10)/6*(x**3)

#*6 for qua, 36 for 1-6 NN
def cpr(val1, val2):
    p1v1 = phi1(val1)
    p2v1 = phi2(val1)
    p3v1 = phi3(val1)
    p1v2 = phi1(val2)
    p2v2 = phi2(val2)
    p3v2 = phi3(val2)
    c11 = p1v1*p1v2
    c12 = (p1v1*p2v2+p2v1*p1v2)/2
    c13 = (p1v1*p3v2+p3v1*p1v2)/2
    c22 = p2v1*p2v2
    c23 = (p2v1*p3v2+p3v1*p2v2)/2
    c33 = p3v1*p3v2

    return np.array([c11, c12, c13, c22, c23, c33])

def ideal_cor(cr_content, mn_content, co_content, Nnn, mode='printNG'):
    ni_content = 1-cr_content-mn_content-co_content
    bond_num = len(Nnn)

    num_crcr = cr_content**2*bond_num
    num_mnmn = mn_content**2*bond_num
    num_coco = co_content**2*bond_num
    num_nini = ni_content**2*bond_num
    num_crco = 2*cr_content*co_content*bond_num
    num_crmn = 2*cr_content*mn_content*bond_num
    num_coni = 2*co_content*ni_content*bond_num
    num_comn = 2*co_content*mn_content*bond_num
    num_crni = 2*cr_content*ni_content*bond_num
    num_mnni = 2*mn_content*ni_content*bond_num

    #*2, 1, -1, -2: Cr, Mn, Co, Ni
    cor_func = (num_crcr*cpr(2,2)
               +num_mnmn*cpr(1,1)
               +num_coco*cpr(-1,-1)
               +num_nini*cpr(-2,-2)
               +num_crco*cpr(2,-1)
               +num_crmn*cpr(2,1)
               +num_comn*cpr(-1,1)
               +num_crni*cpr(2,-2)
               +num_mnni*cpr(1,-2)
               +num_coni*cpr(-1,-2))
    
    if mode == 'print':
        print(f'ideal cor func of Cr{cr_content*100}Co{co_content*100}Ni{ni_content*100}: {cor_func}')
    return cor_func

def cor_func(ind_nNN, ele_list):
    cor_func_n = np.zeros(6)
    for i in ind_nNN:
        a1 = ele_list[i[0]]
        a2 = ele_list[i[1]]
        cor_f = cpr(a1, a2)
        cor_func_n += cor_f
    
    return cor_func_n

def ele_list_gen(cr_c, mn_c, co_c, ni_c, mode = 'randchoice'):
    np.random.seed()

    assert abs(cr_c + mn_c + co_c + ni_c - 1) < 0.001, 'Make sure atomic ratio sums to 1'

    while True:
        if mode == 'randchoice':
            len_cr = randrange(int(cr_c*216),int(cr_c*216)+2)
            len_mn = randrange(int(mn_c*216),int(mn_c*216)+2)
            len_co = randrange(int(co_c*216),int(co_c*216)+2)
        elif mode == 'int':
            len_cr = int(cr_c*216)
            len_mn = int(mn_c*216)
            len_co = int(co_c*216)
        
        len_ni = 216-len_cr-len_mn-len_co
        if abs(len_ni-216*ni_c) <= 1:
            break

    ele_list_raw = np.concatenate([np.zeros(len_cr)+2,np.ones(len_mn),0-np.ones(len_co),-1-np.ones(len_ni)],axis=0)
    np.random.shuffle(ele_list_raw)
    
    return ele_list_raw
