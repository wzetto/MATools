import math
import numpy as np 
import sub_func_ce
from sub_func_ce import abs_dis, find_overlap
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from random import randrange
from IPython.display import clear_output
import os
import pickle
import multiprocessing as mp
import time

# from torch.nn.utils import clip_grad_norm_
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

def draw_3d(ind_raw):
    ind_raw = np.array(ind_raw)
    plt.rcParams["figure.figsize"] = [5, 5]
    # plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ind_raw[:,0], ind_raw[:,1], ind_raw[:,2], alpha = 0.5, c = 'r')
    plt.show()

def ele_list_gen(cr_c, mn_c, co_c, ni_c, num_c, mode = 'randchoice'):
    np.random.seed()

    assert abs(cr_c + mn_c + co_c + ni_c - 1) < 0.001, 'Make sure atomic ratio sum to 1'

    while True:
        if mode == 'randchoice':
            len_cr = randrange(int(cr_c*num_c),int(cr_c*num_c)+2)
            len_mn = randrange(int(mn_c*num_c),int(mn_c*num_c)+2)
            len_co = randrange(int(co_c*num_c),int(co_c*num_c)+2)
        elif mode == 'int':
            len_cr = int(cr_c*num_c)
            len_mn = int(mn_c*num_c)
            len_co = int(co_c*num_c)
        
        len_ni = num_c-len_cr-len_mn-len_co
        if abs(len_ni-num_c*ni_c) <= 1:
            break

    ele_list_raw = np.concatenate([np.zeros(len_cr)+2,np.ones(len_mn),0-np.ones(len_co),-1-np.ones(len_ni)],axis=0)
    np.random.shuffle(ele_list_raw)
    
    return ele_list_raw

def swap_step(action, state,):

    a1 = action[0]
    a2 = action[1]

    state[a2], state[a1] = state[a1], state[a2]

    return state

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created Directory : ", directory)
    else:
        print("Directory already existed : ", directory)
    return directory

class CE:
    def __init__(self, ind_1nn, ind_2nn, ind_3nn, ind_4nn, 
                ind_qua1nn, ind_qua1nn2nn,
                ind_trip1nn, ind_trip1nn2nn, ind_trip1nn2nn3nn,
                ind_raw):

        self.ind_1nn = ind_1nn
        self.ind_2nn = ind_2nn
        self.ind_3nn = ind_3nn
        self.ind_4nn = ind_4nn
        self.ind_qua1nn = ind_qua1nn
        self.ind_qua1nn2nn = ind_qua1nn2nn
        self.ind_trip1nn = ind_trip1nn
        self.ind_trip1nn2nn = ind_trip1nn2nn
        self.ind_trip1nn2nn3nn = ind_trip1nn2nn3nn
        self.ind_raw = ind_raw
        self.sym_oplist = np.array([2, 1, 1, 0, 6, 0, 4, 12, 24])
        self.sym_optri = np.array([6, 3, 1])
        self.sym_opqua = np.array([0, 0, 1, 0, 0, 0, 2, 0, 4])

    #*Normalizaiton by the symmetry operation for each cluster
    def sym_operator(self, cluster, mode='None'):
        if mode == 'None':
            mode_cluster = len(cluster)
            uni, count_uni = np.unique(cluster, return_counts=True)
            len_uni = len(uni)
            count_uni = np.sort(count_uni)
            mode_sym = mode_cluster - len(uni)
            sym_op = self.sym_oplist[mode_sym+mode_cluster-2]

        if mode == 'tri1nn':
            mode_cluster = len(cluster)
            uni, count_uni = np.unique(cluster, return_counts=True)
            len_uni = len(uni)
            mode_sym = mode_cluster - len(uni)
            sym_op = self.sym_optri[mode_sym]
        
        if mode == 'qua1nn':
            uni, count_uni = np.unique(cluster, return_counts=True)
            len_uni = len(uni)
            count_uni = np.sort(count_uni)
            len_count = len(count_uni)
            c_ind = len_uni+len_count+count_uni[-1]-count_uni[0]
            sym_op = self.sym_oplist[c_ind]

        if mode == 'qua1nn2nn':
            uni, count_uni = np.unique(cluster, return_counts=True)
            len_uni = len(uni)
            count_uni = np.sort(count_uni)
            len_count = len(count_uni)
            c_ind = len_uni+len_count+count_uni[-1]-count_uni[0]
            single_list = np.array([2, 6, 8])
            if c_ind in single_list:
                sym_op = self.sym_opqua[c_ind]
            else:
                if c_ind == 4:
                    if len(np.unique(cluster[:2])) == 2:
                        sym_op = 4
                    else:
                        sym_op = 1

                elif c_ind == 7:
                    if len(np.unique(cluster[:2])) == 2 and len(np.unique(cluster[2:4])) == 2:
                        sym_op = 4
                    else:
                        sym_op = 2

        return sym_op

    def phi1(self, x):
        return 2/math.sqrt(10)*x

    def phi2(self, x):
        return -5/3 + 2/3*(x**2)

    def phi3(self, x):
        return -17/30*math.sqrt(10)*x + math.sqrt(10)/6*(x**3)
    
    #*Return the correlation function for each cluster
    def cpr(self, val_list):
        p1l = self.phi1(val_list).reshape(-1, 1)
        p2l = self.phi2(val_list).reshape(-1, 1)
        p3l = self.phi3(val_list).reshape(-1, 1)
        pl = np.concatenate([p1l, p2l, p3l], 1).T
        c_len = len(val_list)
        atom = 1
        for i in range(c_len):
            atom_1 = pl[:, i]
            atom = np.outer(atom_1, atom)

        return atom.flatten()

    def cluster_extra(self, config):
        cpr_1nn = 0
        for i in self.ind_1nn:
            a1, a2 = config[i[0]], config[i[1]]
            cluster = np.array([a1, a2])
            cpr_1nn += self.cpr(cluster)/self.sym_operator(cluster)
        
        cpr_1nn /= len(self.ind_1nn)

        cpr_2nn = 0
        for i in self.ind_2nn:
            a1, a2 = config[i[0]], config[i[1]]
            cluster = np.array([a1, a2])
            cpr_2nn += self.cpr(cluster)/self.sym_operator(cluster)

        cpr_2nn /= len(self.ind_2nn)
        
        cpr_3nn = 0
        for i in self.ind_3nn:
            a1, a2 = config[i[0]], config[i[1]]
            cluster = np.array([a1, a2])
            cpr_3nn += self.cpr(cluster)/self.sym_operator(cluster)

        cpr_3nn /= len(self.ind_3nn)

        cpr_4nn = 0
        for i in self.ind_4nn:
            a1, a2 = config[i[0]], config[i[1]]
            cluster = np.array([a1, a2])
            cpr_4nn += self.cpr(cluster)/self.sym_operator(cluster)

        cpr_4nn /= len(self.ind_4nn)
            
        cpr_tri1nn = 0
        for i in self.ind_trip1nn:
            a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            cluster = np.array([a1, a2, a3])
            cpr_tri1nn += self.cpr(cluster)/self.sym_operator(cluster, mode='tri1nn')

        cpr_tri1nn /= len(self.ind_trip1nn)

        cpr_tri1nn2nn = 0
        for i in self.ind_trip1nn2nn:
            a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            cluster = np.array([a1, a2, a3])
            #*The only operation in this case when in AAB form
            if (a1 == a2 and a1 != a3) or (a1 == a3 and a1 != a2):
                cpr_tri1nn2nn += self.cpr(cluster)/2
            else:
                cpr_tri1nn2nn += self.cpr(cluster)

        cpr_tri1nn2nn /= len(self.ind_trip1nn2nn)

        cpr_tri1nn2nn3nn = 0
        for i in self.ind_trip1nn2nn3nn:
            a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            cluster = np.array([a1, a2, a3])
            cpr_tri1nn2nn3nn += self.cpr(cluster)

        cpr_tri1nn2nn3nn /= len(self.ind_trip1nn2nn3nn)

        #!Consider the operation in qua later
        cpr_qua1nn = 0
        for i in self.ind_qua1nn:
            a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
            cluster = np.array([a1, a2, a3, a4])
            cpr_qua1nn += self.cpr(cluster)/self.sym_operator(cluster, mode='qua1nn')

        cpr_qua1nn /= len(self.ind_qua1nn)

        cpr_qua1nn2nn = 0
        for i in self.ind_qua1nn2nn:
            a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
            cluster = np.array([a1, a2, a3, a4])
            cpr_qua1nn2nn += self.cpr(cluster)/self.sym_operator(cluster, mode='qua1nn2nn')

        cpr_qua1nn2nn /= len(self.ind_qua1nn2nn)

        return np.concatenate([
            cpr_1nn, cpr_2nn, cpr_3nn, cpr_4nn,
            cpr_tri1nn, cpr_tri1nn2nn, cpr_tri1nn2nn3nn,
            cpr_qua1nn,
        ], 0)

    def config_extra(self, num_cell, ind_cr, ind_mn, ind_co, ind_ni):
        ele_list = np.zeros(num_cell)
        overlap_cr = find_overlap(self.ind_raw, ind_cr)
        overlap_mn = find_overlap(self.ind_raw, ind_mn)
        overlap_co = find_overlap(self.ind_raw, ind_co)
        overlap_ni = find_overlap(self.ind_raw, ind_ni)

        ele_list[np.where(overlap_cr)[0]] = 2
        ele_list[np.where(overlap_mn)[0]] = 1
        ele_list[np.where(overlap_co)[0]] = -1
        ele_list[np.where(overlap_ni)[0]] = -2

        cpr_list = self.cluster_extra(ele_list)

        return cpr_list

    def read(self, incar_dir):
        with open(incar_dir) as f:
            input_strip = [s.strip() for s in f.readlines()]
        return input_strip

def t_range(temp_0, t, tau):
    return temp_0*np.exp(-t/tau)

def main(temp):

    sro_list_store = np.zeros((iter_time, 10))
    config = ele_list_gen(1/4, 1/4, 1/4, 1/4, num_c=2048)
    #* Initialize the cluster model
    ce_e = CE(ind_1nn, ind_2nn, ind_3nn, ind_4nn, 
        ind_qua1nn, ind_qua1nn2nn,
        ind_trip1nn, ind_trip1nn2nn, ind_trip1nn2nn3nn,
        ind_raw)
    
    #* Load the lr model
    atom_num = 2048
    date = '20221101'
    pth = f'/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/MATools/CE_MC/runs/demo/{date}'
    lr_name = '/blr.sav'
    clf_ = pickle.load(open(pth+lr_name, 'rb'))

    for i in range(iter_time):
        # weight_config = norm_w(ce_e.cluster_extra(config).reshape(-1,1).T,
        #                 weight_mean, weight_std)
        weight_config  = ce_e.cluster_extra(config).reshape(-1,1).T
        #* NN's prediction
        # weight_config = torch.from_numpy(weight_config.astype(np.float32)).clone().to(device)
        # energy = fc_(weight_config).cpu().detach().numpy()[0,0]*energy_std + energy_mean
        #* LR's prediction
        energy = clf_.predict(weight_config)*energy_std + energy_mean
        energy *= atom_num
        # e_list.append(energy)
        # config_list[iter_time%50] = config
        # e_list_store[iter_time%50] = energy
        #* Extract SRO params of current config..
        sro = sub_func_ce.sro_extra(ind_1nn, config, 0.25, 0.25, 0.25, 0.25)
        sro_list_store[i%10000] = sro

        while True:
            a_ind = randrange(len(ind_1nn))
            action = ind_1nn[a_ind]
            a1, a2 = config[action[0]], config[action[1]]
            if a1 != a2:
                break

        config_ = swap_step(action, config)
        # weight_config_ = norm_w(ce_e.cluster_extra(config_).reshape(-1,1).T,
        #                 weight_mean, weight_std)
        weight_config_ = ce_e.cluster_extra(config_).reshape(-1,1).T
        #* NN's prediction
        # weight_config_ = torch.from_numpy(weight_config_.astype(np.float32)).clone().to(device)
        # energy_ = fc_(weight_config_).cpu().detach().numpy()[0,0]*energy_std + energy_mean
        #* LR's prediction
        energy_ = clf_.predict(weight_config_)*energy_std + energy_mean
        energy_ *= atom_num

        accept = np.min([1, np.exp((energy[0]-energy_[0])/(k_*temp))])
        r_v = np.random.rand()
        if r_v <= accept:
                config = config_
        else:
                config = config

    np.save(pth+f'/sro_{temp}', sro_list_store)
    

if __name__ == '__main__':
    start_ = time.time()

    ind_1nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_1nn.npy')
    ind_2nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_2nn.npy')
    ind_3nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_3nn.npy')
    ind_4nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_4nn.npy')
    ind_qua1nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_qua1nn.npy')
    ind_qua1nn2nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_qua1nn2nn.npy')
    ind_trip1nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_trip1nn.npy')
    ind_trip1nn2nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_trip1nn2nn.npy')
    ind_trip1nn2nn3nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_trip1nn2nn3nn.npy')
    ind_raw = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_2048/ind_raw2048.npy')

    iter_time= 7500
    k_ = 8.617333262e-5
    energy_std, energy_mean = 1, 0
    temp_range = np.array([100, 200, 300, 400, 500, 800, 1000, 1200, 1600, 2400, 3200, 4800])
    try_num = len(temp_range) #* Number of processors

    pool = mp.Pool(processes = try_num)
    pool.map(main, temp_range)
    pool.close()
    pool.join()

    end_ = time.time()
    second_to_hour(end_-start_)