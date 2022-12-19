import math
import numpy as np
from random import randrange

ind_1nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_192/ind_1nn.npy')
ind_2nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_192/ind_2nn.npy')
ind_3nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_192/ind_3nn.npy')
ind_4nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_192/ind_4nn.npy')
ind_5nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_192/ind_5nn.npy')
ind_6nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_192/ind_6nn.npy')

cr_, mn_, co_, ni_ = 1/3, 1/3, 1/3, 1/3
def atom_get():
    return cr_, mn_, co_, ni_

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

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

def ideal_cor(cr_content, mn_content, co_content, Nnn, mode='dontprintbro'):
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
    
    if mode == 'please print man!':
        print(f'ideal cor func of Cr{cr_content*100}Co{co_content*100}Ni{ni_content*100}: {cor_func}')
    return cor_func

ideal_1, ideal_2, ideal_3, ideal_4, ideal_5, ideal_6 = (
    ideal_cor(cr_, mn_, co_, ind_1nn), 
    ideal_cor(cr_, mn_, co_, ind_2nn),
    ideal_cor(cr_, mn_, co_, ind_3nn),
    ideal_cor(cr_, mn_, co_, ind_4nn),
    ideal_cor(cr_, mn_, co_, ind_5nn),
    ideal_cor(cr_, mn_, co_, ind_6nn))

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

    assert abs(cr_c + mn_c + co_c + ni_c - 1) < 0.001, 'Make sure atomic ratio sum to 1'

    while True:
        if mode == 'randchoice':
            len_cr = randrange(int(cr_c*192),int(cr_c*192)+2)
            len_mn = randrange(int(mn_c*192),int(mn_c*192)+2)
            len_co = randrange(int(co_c*192),int(co_c*192)+2)
        elif mode == 'int':
            len_cr = int(cr_c*192)
            len_mn = int(mn_c*192)
            len_co = int(co_c*192)
        
        len_ni = 192-len_cr-len_mn-len_co
        if abs(len_ni-192*ni_c) <= 1:
            break

    ele_list_raw = np.concatenate([np.zeros(len_cr)+2,np.ones(len_mn),0-np.ones(len_co),-1-np.ones(len_ni)],axis=0)
    np.random.shuffle(ele_list_raw)
    
    return ele_list_raw

def cor_func_all(state, mode='abs'):
    cor1 = cor_func(ind_1nn, state) - ideal_1
    cor2 = cor_func(ind_2nn, state) - ideal_2
    cor3 = cor_func(ind_3nn, state) - ideal_3
    cor4 = cor_func(ind_4nn, state) - ideal_4
    cor5 = cor_func(ind_5nn, state) - ideal_5
    cor6 = cor_func(ind_6nn, state) - ideal_6

    cor_ = np.concatenate([cor1, cor2, cor3, cor4, cor5, cor6])
    return np.linalg.norm(cor_)

def swap_step(action, cor_func_n, state, target_val):

    # cor_func_raw = ((abs(cor_func(ind_1nn, state)-ideal_1)
    #                 +abs(cor_func(ind_2nn, state)-ideal_2)
    #                 +abs(cor_func(ind_3nn, state)-ideal_3)
    #                 +abs(cor_func(ind_4nn, state)-ideal_4))).copy()

    cor_func_raw = cor_func_n

    a1 = action[0]
    a2 = action[1]

    state[a2], state[a1] = state[a1], state[a2]

    cor_func_new = cor_func_all(state)

    reward = cor_func_raw - cor_func_new
    
    if cor_func_new < target_val:
        done = True
    else:
        done = False

    return state, reward, cor_func_new, done

# print(ideal_1, ideal_2, ideal_3, ideal_4, ideal_5, ideal_6)

######################################################
crcr = np.array([2,2])
mncr, crmn = np.array([1,2]), np.array([2,1])
cocr, crco = np.array([-1,2]), np.array([2,-1])
nicr, crni = np.array([-2,2]), np.array([2,-2])
mnmn = np.array([1,1])
comn, mnco = np.array([-1,1]), np.array([1,-1])
nimn, mnni = np.array([-2,1]), np.array([1,-2])
coco = np.array([-1,-1])
nico, coni = np.array([-2,-1]), np.array([-1,-2])
nini = np.array([-2,-2])

#* Extracting the 1NN short range order parameter
#* 2, 1, -1, -2: Cr, Mn, Co, Ni
#* a_crcr, a_mnmn, a_coco, a_nini, a_mncr, a_cocr, a_nicr, a_comn, a_nimn, a_nico
def sro_extra(ind_nNN, config, cr_, mn_, co_, ni_):
    assert cr_+mn_+co_+ni_ == 1, print('?1?')
    # num_cr, num_mn, num_co, num_ni = np.zeros(4)
    n_crcr, n_mncr, n_cocr, n_nicr, n_mnmn, n_comn, n_nimn, n_coco, n_nico, n_nini = np.zeros(10)
    # n_crmn, n_crco, n_crni, n_mnco, n_mnni, n_coni = np.zeros(6)
    len_all = len(ind_nNN)

    for i in ind_nNN:
        #* Returns the sorted atomic pair
        pair_list = np.sort(np.array([config[i[0]], config[i[1]]]))

        if np.linalg.norm(pair_list-crcr) == 0:
            # num_cr += 1
            n_crcr += 1

        elif np.linalg.norm(pair_list-mnmn) == 0:
            # num_mn += 2
            n_mnmn += 1

        elif np.linalg.norm(pair_list-coco) == 0:
            # num_co += 2
            n_coco += 1

        elif np.linalg.norm(pair_list-nini) == 0:
            # num_ni += 2
            n_nini += 1

        elif np.linalg.norm(pair_list-mncr) == 0:
            # num_mn += 1
            # num_cr += 1
            n_mncr += 1

        elif np.linalg.norm(pair_list-cocr) == 0:
            # num_co += 1
            # num_cr += 1
            n_cocr += 1

        elif np.linalg.norm(pair_list-nicr) == 0:
            # num_ni += 1
            # num_cr += 1
            n_nicr += 1

        elif np.linalg.norm(pair_list-comn) == 0:
            # num_co += 1
            # num_mn += 1
            n_comn += 1

        elif np.linalg.norm(pair_list-nimn) == 0:
            # num_ni += 1
            # num_mn += 1
            n_nimn += 1

        elif np.linalg.norm(pair_list-nico) == 0:
            # num_ni += 1
            # num_co += 1
            n_nico += 1

    # a_crcr = 1 - n_crcr/num_cr/(cr_)
    # a_mnmn = 1 - n_mnmn/num_mn/(mn_)
    # a_coco = 1 - n_coco/num_co/(co_)
    # a_nini = 1 - n_nini/num_ni/(ni_)
    # a_mncr = 1 - n_mncr/num_mn/(cr_)
    # a_crmn = 1 - n_crmn/num_cr/(mn_)
    # a_cocr = 1 - n_cocr/num_co/(cr_)
    # a_crco = 1 - n_crco/num_cr/(co_)
    # a_nicr = 1 - n_nicr/num_ni/(cr_)
    # a_crni = 1 - n_crni/num_cr/(ni_)
    # a_comn = 1 - n_comn/num_co/(mn_)
    # a_mnco = 1 - n_mnco/num_mn/(co_)
    # a_nimn = 1 - n_nimn/num_ni/(mn_)
    # a_mnni = 1 - n_mnni/num_mn/(ni_)
    # a_nico = 1 - n_nico/num_ni/(co_) 
    # a_coni = 1 - n_coni/num_co/(ni_) 

    ''' 
    This equation returns the overall probability 
    of finding i-j pair at distance r.
    '''
    a_crcr = 1 - n_crcr/len_all/(2*cr_*cr_)
    a_mnmn = 1 - n_mnmn/len_all/(2*mn_*mn_)
    a_coco = 1 - n_coco/len_all/(2*co_*co_)
    a_nini = 1 - n_nini/len_all/(2*ni_*ni_)

    a_mncr = 1 - n_mncr/len_all/(2*mn_*cr_)
    a_cocr = 1 - n_cocr/len_all/(2*co_*cr_)
    a_nicr = 1 - n_nicr/len_all/(2*ni_*cr_)
    a_comn = 1 - n_comn/len_all/(2*co_*mn_)
    a_nimn = 1 - n_nimn/len_all/(2*ni_*mn_)
    a_nico = 1 - n_nico/len_all/(2*ni_*co_)

    # len_check = (n_crcr+n_mnmn+n_coco+n_nini+n_mncr+n_cocr
    #     +n_nicr+n_comn+n_nimn+n_nico)

    return np.array([
        a_crcr, a_mnmn, a_coco, a_nini, 
        a_mncr, a_cocr, a_nicr, a_comn,
        a_nimn, a_nico, 
    ])