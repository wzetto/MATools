import numpy as np
import math
from random import randrange
from itertools import product, combinations
from sklearn.model_selection import train_test_split
# simp_struc_gen
# gen_crconi_sfe
# nn_cor_func_new
# nn_cor_func_inter
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

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

def phi0(x):
    return 1

def phi1(x):
    return math.sqrt(3/2)*x

def phi2(x):
    return math.sqrt(2)*(3/2*(x**2) - 1)

def cpr(val1, val2):
    basis_1 = [phi0(val1), phi1(val1), phi2(val1)]
    basis_2 = [phi0(val2), phi1(val2), phi2(val2)]
    cor_func = np.sum(np.outer(basis_1, basis_2))
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
                cor_func = cpr(0,1)/2
            elif pair_overlap == [1,0,1]:
                cor_func = cpr(0,-1)/2
            elif pair_overlap == [0,2,0]:
                cor_func = cpr(1,1)
            elif pair_overlap == [0,1,1]:
                cor_func = cpr(-1,1)/2
            elif pair_overlap == [0,0,2]:
                cor_func = cpr(-1,-1)
            
            cor_fun += cor_func
  
    return cor_fun

def simp_struc_gen(mode = 'simp'):
    x_u = math.sqrt(2)/2
    y_u = math.sqrt(3/2)
    z_u = math.sqrt(1/3)

    layer_A1 = np.array([
        #From x to y
        [0,0,0],[0,y_u,0],[0,2*y_u,0],[0,3*y_u,0],[0,4*y_u,0],[0,5*y_u,0],
        [x_u/2,y_u/2,0],[x_u/2,3*y_u/2,0],[x_u/2,5*y_u/2,0],[x_u/2,7*y_u/2,0],[x_u/2,9*y_u/2,0],[x_u/2,11*y_u/2,0],
        [x_u,0,0],[x_u,y_u,0],[x_u,2*y_u,0],[x_u,3*y_u,0],[x_u,4*y_u,0],[x_u,5*y_u,0],
        [3*x_u/2,y_u/2,0],[3*x_u/2,3*y_u/2,0],[3*x_u/2,5*y_u/2,0],[3*x_u/2,7*y_u/2,0],[3*x_u/2,9*y_u/2,0],[3*x_u/2,11*y_u/2,0],
        [2*x_u,0,0],[2*x_u,y_u,0],[2*x_u,2*y_u,0],[2*x_u,3*y_u,0],[2*x_u,4*y_u,0],[2*x_u,5*y_u,0],
        [5*x_u/2,y_u/2,0],[5*x_u/2,3*y_u/2,0],[5*x_u/2,5*y_u/2,0],[5*x_u/2,7*y_u/2,0],[5*x_u/2,9*y_u/2,0],[5*x_u/2,11*y_u/2,0],
    ])
    layer_B1 = layer_A1 + np.array([0,1/3*y_u,z_u])
    layer_C1 = layer_B1 + np.array([0,1/3*y_u,z_u])
    for i in layer_C1:
        if i[0] == x_u/2 or i[0] == 3*x_u/2 or i[0] == 5*x_u/2:
            i[1] = i[1] - y_u

    layer_A2 = layer_A1 + np.array([0,0,3*z_u])
    layer_B2 = layer_B1 + np.array([0,0,3*z_u])
    layer_C2 = layer_C1 + np.array([0,0,3*z_u])

    layer_A3 = layer_A2 + np.array([0,0,3*z_u])
    layer_B3 = layer_B2 + np.array([0,0,3*z_u])
    layer_C3 = layer_C2 + np.array([0,0,3*z_u])

    layer_A4 = layer_A3 + np.array([0,0,3*z_u])
    layer_B4 = layer_B3 + np.array([0,0,3*z_u])
    layer_C4 = layer_C3 + np.array([0,0,3*z_u])
    if mode == 'normal':
        norm_stack = np.concatenate([layer_A1, layer_B1, layer_C1,
                                     layer_A2, layer_B2, layer_C2,
                                     layer_A3, layer_B3, layer_C3,
                                     layer_A4, layer_B4, layer_C4])
        return norm_stack
    
    if mode == 'simp':
        norm_stack_simp = np.concatenate([layer_A1, layer_B1, layer_C1,
                                          layer_A2, layer_B2, layer_C2])
        return norm_stack_simp

def gen_crconi_sfe(cr_content, co_content, norm_pos):
    norm_raw = norm_pos.copy()
    
    norm_raw_1f, norm_raw_1s = [], []
    tuple_len = [i for i in range(len(norm_raw))]
    train_size = int(len(tuple_len)*cr_content) + randrange(2)
    co_content = co_content/(1-cr_content)
    norm_raw1, norm_raw23 = train_test_split(tuple_len, test_size = len(tuple_len) - train_size, train_size = train_size)
    train_size_sub = int(len(norm_raw23)*co_content) + randrange(2)
    norm_raw2, norm_raw3 = train_test_split(norm_raw23, test_size = len(norm_raw23) - train_size_sub, train_size = train_size_sub)
    
    cr_position = np.array([norm_raw[i] for i in norm_raw1])
    co_position = np.array([norm_raw[i] for i in norm_raw2])
    ni_position = np.array([norm_raw[i] for i in norm_raw3])
    
    ind_raw = np.concatenate([cr_position, co_position, ni_position], axis = 0)
    
    return cr_position, co_position, ni_position, ind_raw

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
                cor_func = cpr(0,1)/2
            elif pair_overlap == [1,0,1]:
                cor_func = cpr(0,-1)/2
            elif pair_overlap == [0,2,0]:
                cor_func = cpr(1,1)
            elif pair_overlap == [0,1,1]:
                cor_func = cpr(-1,1)/2
            elif pair_overlap == [0,0,2]:
                cor_func = cpr(-1,-1)
            
            cor_fun += cor_func
    
    return cor_fun

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
               +num_crco*cpr(0,1)/2
               +num_crni*cpr(0,-1)/2
               +num_coni*cpr(1,-1)/2)
    if mode == 'dont print bro':
        return cor_func
    elif mode == 'print bro':
        print(f'ideal cor func of Cr{int(cr_content*100)}Co{int(co_content*100)}Ni{int(ni_content*100)}: {cor_func}')
        return cor_func