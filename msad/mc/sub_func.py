import math
import numpy as np
from random import randrange

ind_1nn = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_1nn.npy')
ind_2nn = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_2nn.npy')
ind_3nn = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_3nn.npy')
ind_4nn = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_4nn.npy')
ind_5nn = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_5nn.npy')
ind_6nn = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_6nn.npy')
ind_tri1 = np.load('G:/マイドライブ/project/SQS_drl/fcc_108/ind_tri1nn.npy')
# ele_demo_20 = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/elelist_3333_20.npy')[:10]
# ele_demo_30 = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/elelist_3333_30.npy')[:10]
# ele_demo_40 = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/elelist_3333_40.npy')[:10]
# ele_demo = np.concatenate([ele_demo_40, ele_demo_20, ele_demo_30], axis=0)
# len_demo = len(ele_demo)
cr_, co_, ni_ = 0.5, 0.15, 0.35

def atom_get():
    return cr_, co_, ni_

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

def phi1(x):
    return math.sqrt(3/2)*x

def phi2(x):
    return math.sqrt(2)*(3/2*(x**2) - 1)

def cpr(val1, val2):
    cor_func = (phi1(val1)*phi1(val2)
            +phi1(val1)*phi2(val2)
            +phi2(val1)*phi1(val2)
            +phi2(val1)*phi2(val2))

    return cor_func

def ideal_cor(cr_content, co_content, Nnn, mode = 'Dont print bro'):
    ni_content = 1-cr_content-co_content
    bond_num = len(Nnn)

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
    if mode == 'please print man!':
        print(f'ideal cor func of Cr{cr_content*100}Co{co_content*100}Ni{ni_content*100}: {cor_func}')
    return cor_func

def ideal_tri1(cr, co, ind_tri):
    ni = 1 - cr- co
    n_tri = len(ind_tri)
    i_co2ni = ni*co*co*3*n_tri
    i_cr3 = (cr**3)*n_tri
    return np.array([i_co2ni, i_cr3])

ideal_1, ideal_2, ideal_3, ideal_4, ideal_5, ideal_6, i_co2ni, i_cr3 = (ideal_cor(cr_, co_, ind_1nn),
                                    ideal_cor(cr_, co_, ind_2nn),
                                    ideal_cor(cr_, co_, ind_3nn),
                                    ideal_cor(cr_, co_, ind_4nn),
                                    ideal_cor(cr_, co_, ind_5nn),
                                    ideal_cor(cr_, co_, ind_6nn),
                                    ideal_tri1(cr_, co_, ind_tri1)[0],
                                    ideal_tri1(cr_, co_, ind_tri1)[1])

def cor_func(ind_nNN, ele_list):
    cor_func_n = 0
    for i in ind_nNN:
        phi1_1 = math.sqrt(3/2)*ele_list[i[0]]
        phi1_2 = math.sqrt(2)*(3/2*(ele_list[i[0]]**2) - 1)
        phi2_1 = math.sqrt(3/2)*ele_list[i[1]]
        phi2_2 = math.sqrt(2)*(3/2*(ele_list[i[1]]**2) - 1)
        cor_func_n += phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2
    return cor_func_n

def tri_count(ind_tri, ele_list):
    count_cr3, count_co2ni = 0, 0
    for ind in ind_tri:
        tri_l = np.sort(np.array([ele_list[i] for i in ind]))
        if np.linalg.norm(tri_l - np.array([-1, 1, 1])) == 0:
            count_co2ni += 1
        elif np.linalg.norm(tri_l - np.array([0, 0, 0])) == 0:
            count_cr3 += 1

    return np.array([count_co2ni, count_cr3])

def ele_list_gen(cr_content, co_content, ni_content, mode='int'):
    np.random.seed()
    # if iter <= len_demo:
    #     return ele_demo[iter]
    assert abs(cr_content+co_content+ni_content-1)<0.001, 'Make sure atomic ratio sum to 1'

    while True:
        if mode == 'randchoice':
            len_cr = randrange(int(cr_content*108),int(cr_content*108)+2)
            len_co = randrange(int(co_content*108),int(co_content*108)+2)
        elif mode == 'int':
            len_cr = int(cr_content*108)
            len_co = int(co_content*108)
        
        len_ni = 108-len_cr-len_co
        if abs(len_ni-108*ni_content) <= 1:
            break

    ele_list_raw = np.concatenate([np.zeros(len_cr),np.ones(len_co),0-np.ones(len_ni)],axis=0)
    np.random.shuffle(ele_list_raw)
    
    return ele_list_raw

def cor_func_all(state, mode='abs'):
    n_co2ni, n_cr3 = tri_count(ind_tri1, state)
    if mode == 'abs':
        return (abs(cor_func(ind_1nn, state)-ideal_1)
                +abs(cor_func(ind_2nn, state)-ideal_2)
                +abs(cor_func(ind_3nn, state)-ideal_3)
                +abs(cor_func(ind_4nn, state)-ideal_4)
                +abs(cor_func(ind_5nn, state)-ideal_5)
                +abs(cor_func(ind_6nn, state)-ideal_6)
                +abs(n_co2ni - i_co2ni)
                +abs(n_cr3 - i_cr3))
    
    elif mode == 'seperate':
        return [cor_func(ind_1nn, state), 
        cor_func(ind_2nn, state), 
        cor_func(ind_3nn, state), 
        cor_func(ind_4nn, state),
        cor_func(ind_5nn, state),
        cor_func(ind_6nn, state),
        n_co2ni,
        n_cr3]

def swap_step(action, cor_func_n, state, target_val):

    # cor_func_raw = ((abs(cor_func(ind_1nn, state)-ideal_1)
    #                 +abs(cor_func(ind_2nn, state)-ideal_2)
    #                 +abs(cor_func(ind_3nn, state)-ideal_3)
    #                 +abs(cor_func(ind_4nn, state)-ideal_4))).copy()

    cor_func_raw = cor_func_n

    a1 = action[0]
    a2 = action[1]

    state[a2], state[a1] = state[a1], state[a2]
    n_co2ni, n_cr3 = tri_count(ind_tri1, state)

    cor_func_new = ((abs(cor_func(ind_1nn, state)-ideal_1)
                    +abs(cor_func(ind_2nn, state)-ideal_2)
                    +abs(cor_func(ind_3nn, state)-ideal_3)
                    +abs(cor_func(ind_4nn, state)-ideal_4)
                    +abs(cor_func(ind_5nn, state)-ideal_5)
                    +abs(cor_func(ind_6nn, state)-ideal_6)
                    +abs(n_co2ni - i_co2ni)
                    +abs(n_cr3 - i_cr3))).copy()

    reward = cor_func_raw - cor_func_new
    
    if cor_func_new < target_val:
        done = True
    else:
        done = False

    # if reward > 0.001:
    #     reward = reward*2

    # if reward == 0:
    #     reward = -np.random.rand()

    return state, reward, cor_func_new, done

# print(ideal_1, ideal_2, ideal_3, ideal_4, ideal_5, ideal_6, i_co2ni, i_cr3)