from numba import cuda
import numpy as np
import math
import time

def phi1(x):
    return math.sqrt(3/2)*x

def phi2(x):
    return math.sqrt(2)*(3/2*(x**2) - 1)

def cpr(val1, val2):
    basis_1 = cuda.to_device(np.array([phi1(val1), phi2(val1)]))
    basis_2 = cuda.to_device(np.array([phi1(val2), phi2(val2)]))
    cor_func = np.sum(np.outer(basis_1, basis_2))
    return cor_func

def cor_func(ind_nNN, ele_list):
    cor_func_n = 0
    for i in ind_nNN:
        cor_func_n += cpr(ele_list[i[0]], ele_list[i[1]])
    return cor_func_n

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))

# !However, when targeting cuda all Numpy functions that allocate memory (
# !including np.zeros) are disabled.

@cuda.jit
def l2_list_action(a1_value, ele_list_s, cor_1_1, cor_1_2, 
                   cor_2, cor_3_1, cor_3_2, cor_4):

    b_ind = cuda.threadIdx.x
    t_ind = cuda.blockIdx.x #*a1
    if t_ind < 108:

        if b_ind < 108:
            ind_nn = ind_1nn_1
            iter_time = 225
            cor_nn = cor_1_1
            b_ind_true = b_ind
        elif 108 <= b_ind < 216:
            ind_nn = ind_1nn_2
            iter_time = 225
            cor_nn = cor_1_2
            b_ind_true = b_ind - 108
        elif 216 <= b_ind < 324:
            ind_nn = ind_2nn
            iter_time = 216
            cor_nn = cor_2
            b_ind_true = b_ind - 216
        elif 324 <= b_ind < 432:
            ind_nn = ind_3nn_1
            iter_time = 300
            cor_nn = cor_3_1
            b_ind_true = b_ind - 324
        elif 432 <= b_ind < 540:
            ind_nn = ind_3nn_2
            iter_time = 300
            cor_nn = cor_3_2
            b_ind_true = b_ind - 432
        elif 540 <= b_ind < 648:
            ind_nn = ind_4nn
            iter_time = 288
            cor_nn = cor_4
            b_ind_true = b_ind - 540
         
        ideal_nn = 0
        cor_func_raw = abs(cor_nn - ideal_nn)
        cor_new = 0
        for i in range(iter_time):
            if ind_nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind_true]
                val2 = ele_list_s[ind_nn[i][1]]
            elif ind_nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind_true]
                val1 = ele_list_s[ind_nn[i][0]]   
            elif ind_nn[i][0] == b_ind_true:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_nn[i][1]]
            elif ind_nn[i][1] == b_ind_true:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_nn[i][0]]
            else:             
                val1 = ele_list_s[ind_nn[i][0]]
                val2 = ele_list_s[ind_nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_new += cor_func
        
        cor_func_new = abs(cor_new - ideal_nn)
        swap_reward = cor_func_new - cor_func_raw
        a1_value[t_ind, b_ind_true] += swap_reward

def main(iter_time):
    block = 648
    thread = 128

    ele_list, a1_list = [], []
    for _ in range(iter_time):
        # ?rawlist
        np.random.seed()

        raw_list = np.zeros(108, dtype=np.float32)
        raw_list[:36] = 0.
        raw_list[36:72] = 1.
        raw_list[72:] = -1.

        a1_value = np.zeros((108, 108), dtype=np.float32)
        a1_value = cuda.to_device(a1_value)
        # !print(raw_list)
        np.random.shuffle(raw_list)

        cor_1_1 = cor_func(ind_1nn_1, raw_list)
        cor_1_2 = cor_func(ind_1nn_2, raw_list)
        cor_2 = cor_func(ind_2nn, raw_list)
        cor_3_1 = cor_func(ind_3nn_1, raw_list)
        cor_3_2 = cor_func(ind_3nn_2, raw_list)
        cor_4 = cor_func(ind_4nn, raw_list)
        # !print(raw_list)

        raw_list = cuda.to_device(raw_list)
        l2_list_action[block, thread](a1_value, raw_list, cor_1_1, cor_1_2, 
                                      cor_2, cor_3_1, cor_3_2, cor_4)
        cuda.synchronize()
        
        # !print(raw_list)
        a1_value = a1_value.copy_to_host()
        # print(a1_value, np.max(a1_value), np.min(a1_value))
        a1_value = np.sum(a1_value, axis=1)
        
        raw_list = raw_list.copy_to_host().tolist()
        best_a1 = np.where(a1_value == np.max(a1_value))[0][0]
        ele_list.append(raw_list)
        a1_list.append(best_a1)

        if _%5000 == 0:
            print(f'iter: {iter_time}')
        
    return a1_list, ele_list

if __name__ == '__main__':
    start_ = time.time()
    ind_1nn = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/ind_1nn.npy')
    ind_2nn = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/ind_2nn.npy')
    ind_3nn = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/ind_3nn.npy')
    ind_4nn = np.load('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/ind_4nn.npy')
    ind_1nn_1, ind_1nn_2 = ind_1nn[:225], ind_1nn[225:]
    ind_3nn_1, ind_3nn_2 = ind_3nn[:300], ind_3nn[300:]

    a1_list, ele_list = main(iter_time = 150000)
    np.save('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/cuda_a1list_l2_train.npy', a1_list)
    np.save('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/cuda_elelist_l2_train.npy', ele_list)
    print('train finish!')
    
    a1_list, ele_list = main(iter_time = 30000)
    np.save('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/cuda_a1list_l2_test.npy', a1_list)
    np.save('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/cuda_elelist_l2_test.npy', ele_list)

    # print(a1_list)
    second_to_hour(time.time() - start_)
