import numpy as np
from itertools import combinations
from sub_func_ce import abs_dis, find_overlap
import multiprocessing as mp
import json 

''' 
Atoms within primary lattice, {i} could be chosen as multi-processing
operator
'''
# for i in range(32): #! <- this thing

def dis_expand(dis_list, ijkl):
    '''
    i,j,k,l -> a1,a2,a3,a4
    0: [(a1, a2) -> dis1, (a1, a3) -> dis2, (a1, a4) -> dis3]
    1: [(a2, a1) -> dis1, (a2, a3) -> dis4, (a2, a4) -> dis5]
    2: [(a3, a1) -> dis2, (a3, a2) -> dis4, (a3, a4) -> dis6]
    3: [(a4, a1) -> dis3, (a4, a2) -> dis5, (a4, a3) -> dis6]
    '''

    dis1, dis2, dis3, dis4, dis5, dis6 = dis_list
    expand_list =  np.sum(np.array([ 
        [dis1, dis2, dis3],
        [dis1, dis4, dis5],
        [dis2, dis4, dis6],
        [dis3, dis5, dis6],
    ]), axis=1)

    #* Return atomic index in PBC config follows an ascending order.
    return ijkl[expand_list.argsort()].tolist()

def main(i):
    ind_quapbc, ind_quaraw = {}, {}
    print(f'Start {i}')
    #* Create empty list for PBC and RAW dicts.
    for type_ in type_cluster_list:
        ind_quapbc[f'{type_}_pbc'] = []
        ind_quaraw[f'{type_}_raw'] = []

    for j, k, l in combinations(range(len_c), 3):
        
        ind_list = np.array([i,j,k,l])
        #* Number of atoms within the primary lattice.
        len_inlat = len(np.where(ind_list<32)[0])
        if 1 <= len_inlat <= 4 and len(np.unique(ind_list))==4:
            #* Atom species
            a1 = ind_pbc[i]
            a2 = ind_pbc[j]
            a3 = ind_pbc[k]
            a4 = ind_pbc[l]
            dis1 = abs_dis(a1, a2, 0)
            dis2 = abs_dis(a1, a3, 0)
            dis3 = abs_dis(a1, a4, 0)
            dis4 = abs_dis(a2, a3, 0)
            dis5 = abs_dis(a2, a4, 0)
            dis6 = abs_dis(a3, a4, 0)
            dis_cluster = np.array([dis1, dis2, dis3, dis4, dis5, dis6])
            #* Return the index 
            dis_cluster_embed = dis_expand(dis_cluster, ind_list)

            ''' 
            Determine the type of quadruplets;
            {ideal_dis} is the target cluster, R=Nx6;
            {dis_list} is the distance of {nn1 -> nn6}, R=1x6;
            {dis_cluster} is the features (list of bound length) of current cluster, R=1x6;
            '''

            ind_cluster = np.where(np.linalg.norm(np.sort(dis_cluster)-ideal_dis, axis=1)<0.001)[0]
            if len(ind_cluster) == 1: #* which indicates the cluster must be one of the targets.
                ind_cluster = ind_cluster[0]
                type_cluster = type_cluster_list[ind_cluster]
                if 1 <= len_inlat <= 3:
                    ind_quapbc[f'{type_cluster}_pbc'].append(dis_cluster_embed)
                elif len_inlat == 4:
                    ind_quaraw[f'{type_cluster}_raw'].append(dis_cluster_embed)

    return ind_quapbc, ind_quaraw

if __name__ == '__main__':
    #* Value of NN
    dis_list = []
    ind_raw = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_raw32.npy')
    for i, j in combinations(range(32), 2):
        dis_list.append(abs_dis(ind_raw[i], ind_raw[j], 0))

    #* Effective distances.
    dis_list = np.sort(np.unique(dis_list))[:4].reshape(-1,1)

    ind_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_raw_pbc32.npy')
    ind_raw = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_raw32.npy')
    pth = '/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/'
    qua_dislist = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/qualist4nn_32.npy')
    len_c = len(ind_pbc)
    ideal_dis = qua_dislist

    type_cluster_list = []
    #* Extract the corresponding cluster's name within ideal_dis list.
    for i in range(len(ideal_dis)):
        ideal_dis_res = ideal_dis[i]-dis_list
        len_type_list = []
        for i in range(ideal_dis_res.shape[1]):
            len_type_list.append(np.where(ideal_dis_res[:,i]==0)[0][0]+1)
        t1, t2, t3, t4, t5, t6 = len_type_list
        cluster_type = f'{t1}nn{t2}nn{t3}nn{t4}nn{t5}nn{t6}nn'
        type_cluster_list.append(cluster_type)

    #* Index of cluster
    ind_quapbc, ind_quaraw = {}, {}
    #* Create empty list for PBC and RAW dicts.
    for type_ in type_cluster_list:
        ind_quapbc[f'{type_}_pbc'] = []
        ind_quaraw[f'{type_}_raw'] = []

    #* Multi-processing part
    try_num = 20
    pool = mp.Pool(processes = try_num)
    outputs = pool.map(main, range(32))
    pool.close()
    pool.join()

    for output in outputs:
        for type_ in type_cluster_list:
            ind_quapbc[f'{type_}_pbc'] += output[0][f'{type_}_pbc']
            ind_quaraw[f'{type_}_raw'] += output[1][f'{type_}_raw']
    
    pth_sav = '/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/MATools/CE_MC/runs/demo/20221216_msadGA/'
    # open a file in write mode
    with open(f'{pth_sav}ind_quapbc.json', 'w') as f:
    # convert the dictionary to a JSON string and write it to the file
        json.dump(ind_quapbc, f)

    with open(f'{pth_sav}ind_quaraw.json', 'w') as f:
    # convert the dictionary to a JSON string and write it to the file
        json.dump(ind_quaraw, f)