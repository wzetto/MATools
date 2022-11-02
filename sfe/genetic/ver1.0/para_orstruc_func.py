import numpy as np
from itertools import product, combinations
import math

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

def phi0(x):
    return 1

def phi1(x):
    return math.sqrt(3/2)*x

def phi2(x):
    return math.sqrt(2)*(3/2*(x**2) - 1)

def find_overlap(A, B):

    if not A.dtype == B.dtype:
        raise TypeError("A and B must have the same dtype")
    if not A.shape[1:] == B.shape[1:]:
        raise ValueError("the shapes of A and B must be identical apart from "
                         "the row dimension")

    # reshape A and B to 2D arrays. force a copy if neccessary in order to
    # ensure that they are C-contiguous.
    A = np.ascontiguousarray(A.reshape(A.shape[0], -1))
    B = np.ascontiguousarray(B.reshape(B.shape[0], -1))

    # void type that views each row in A and B as a single item
    t = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))

    # use in1d to find rows in A that are also in B
    return np.in1d(A.view(t), B.view(t))

def overlap_count(A, B):
    num = len([i for i in find_overlap(A, B) if i == True])
    return num

def cpr(val1, val2):
    basis_1 = [phi1(val1), phi2(val1)]
    basis_2 = [phi1(val2), phi2(val2)]
    cor_func = np.sum(np.outer(basis_1, basis_2))
    return cor_func

def ideal_cor_func(cr_content, co_content, bond_num, mode='dont print bro'):
    ni_content = 1-cr_content-co_content
    
    num_crcr = cr_content**2*bond_num
    num_coco = co_content**2*bond_num
    num_nini = ni_content**2*bond_num
    num_crco = 2*cr_content*co_content*bond_num
    num_coni = 2*co_content*ni_content*bond_num
    num_crni = 2*cr_content*ni_content*bond_num
    
    cor_func = (num_crcr*cpr(0,0)
               +num_coco*cpr(1,1)
               +num_nini*cpr(-1,-1)
               +num_crco*cpr(0,1)
               +num_crni*cpr(0,-1)
               +num_coni*cpr(1,-1))
    if mode == 'dont print bro':
        return cor_func
    elif mode == 'print bro':
        print(f'ideal cor func of Cr{cr_content*100}Co{co_content*100}Ni{ni_content*100}: {cor_func}')
        return cor_func

def nn_cor_func_new(cr_position, co_position, ni_position, ind, bond_dis, threshold):
    pair_list, cor_fun = [], 0
    for i1, i2 in combinations(ind, 2):
        if (abs_dis(i1, i2, bond_dis) < threshold):
            pair = np.array([i1, i2])
            len_cr_bond = overlap_count(pair, cr_position)
            len_co_bond = overlap_count(pair, co_position)
            len_ni_bond = overlap_count(pair, ni_position)

            pair_overlap = [len_cr_bond, len_co_bond, len_ni_bond]

            if pair_overlap == [2,0,0]:
                cor_func = cpr(0,0)
            elif pair_overlap == [1,1,0]:
                cor_func = cpr(0,1)
            elif pair_overlap == [1,0,1]:
                cor_func = cpr(0,-1)
            elif pair_overlap == [0,2,0]:
                cor_func = cpr(1,1)
            elif pair_overlap == [0,1,1]:
                cor_func = cpr(-1,1)
            elif pair_overlap == [0,0,2]:
                cor_func = cpr(-1,-1)
            
            cor_fun += cor_func
    
    return cor_fun

def nn_cor_func_inter(cr_position, co_position, ni_position, ind, bond_dis, layer_bound, threshold):
    pair_list, cor_fun = [], 0
    ind_inter = np.array([ind_i for ind_i in ind if (ind_i[2] > layer_bound[0] and ind_i[2] < layer_bound[1])])
    for i1, i2 in combinations(ind_inter, 2):
        if (abs_dis(i1, i2, bond_dis) < threshold):
            pair = np.array([i1, i2])
            len_cr_bond = overlap_count(pair, cr_position)
            len_co_bond = overlap_count(pair, co_position)
            len_ni_bond = overlap_count(pair, ni_position)

            pair_overlap = [len_cr_bond, len_co_bond, len_ni_bond]

            if pair_overlap == [2,0,0]:
                cor_func = cpr(0,0)
            elif pair_overlap == [1,1,0]:
                cor_func = cpr(0,1)
            elif pair_overlap == [1,0,1]:
                cor_func = cpr(0,-1)
            elif pair_overlap == [0,2,0]:
                cor_func = cpr(1,1)
            elif pair_overlap == [0,1,1]:
                cor_func = cpr(-1,1)
            elif pair_overlap == [0,0,2]:
                cor_func = cpr(-1,-1)
            
            cor_fun += cor_func
    
    return cor_fun