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
def list_action_1_1(a1_value, ele_list_s, cor_1_1):

    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        # ?cor_1_1
        b_ind = cuda.blockIdx.x #*a1

        ideal_1 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_1_1 - ideal_1)
        # !shuffle array

        # !PROBLEM
        cor_1_1n = 0

        for i in range(225):
            if ind_1nn_1[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_1nn_1[i][1]]
            elif ind_1nn_1[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_1nn_1[i][0]]   
            elif ind_1nn_1[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_1nn_1[i][1]]
            elif ind_1nn_1[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_1nn_1[i][0]]
            else:             
                val1 = ele_list_s[ind_1nn_1[i][0]]
                val2 = ele_list_s[ind_1nn_1[i][1]]
            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_1_1n += cor_func
            cor_func_new = abs(cor_1_1n - ideal_1)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

@cuda.jit
def list_action_1_2(a1_value, ele_list_s, cor_1_2):

    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        # ?cor_1_2
        b_ind = cuda.blockIdx.x #*a1
        ideal_1 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_1_2 - ideal_1)
        cor_1_2n = 0

        for i in range(225):
            if ind_1nn_2[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_1nn_2[i][1]]
            elif ind_1nn_2[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_1nn_2[i][0]]   
            elif ind_1nn_2[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_1nn_2[i][1]]
            elif ind_1nn_2[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_1nn_2[i][0]]
            else:             
                val1 = ele_list_s[ind_1nn_2[i][0]]
                val2 = ele_list_s[ind_1nn_2[i][1]]
            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_1_2n += cor_func
            cor_func_new = abs(cor_1_2n - ideal_1)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

@cuda.jit
def list_action_2(a1_value, ele_list_s, cor_2):

    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        # ?cor_2
        b_ind = cuda.blockIdx.x #*a1
        ideal_2 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_2 - ideal_2)
        cor_2n = 0

        for i in range(216):
            if ind_2nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_2nn[i][1]]
            elif ind_2nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_2nn[i][0]]   
            elif ind_2nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_2nn[i][1]]
            elif ind_2nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_2nn[i][0]]
            else:             
                val1 = ele_list_s[ind_2nn[i][0]]
                val2 = ele_list_s[ind_2nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_2n += cor_func
            cor_func_new = abs(cor_2n - ideal_2)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

@cuda.jit
def list_action_3_1(a1_value, ele_list_s, cor_3_1):

    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        b_ind = cuda.blockIdx.x #*a1
        ideal_3 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_3_1 - ideal_3)
        cor_3_1n = 0

        for i in range(300):
            if ind_3nn_1[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_3nn_1[i][1]]
            elif ind_3nn_1[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_3nn_1[i][0]]   
            elif ind_3nn_1[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_3nn_1[i][1]]
            elif ind_3nn_1[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_3nn_1[i][0]]
            else:             
                val1 = ele_list_s[ind_3nn_1[i][0]]
                val2 = ele_list_s[ind_3nn_1[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_3_1n += cor_func
            cor_func_new = abs(cor_3_1n - ideal_3)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

@cuda.jit
def list_action_3_2(a1_value, ele_list_s, cor_3_2):

    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        b_ind = cuda.blockIdx.x #*a1
        ideal_3 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_3_2 - ideal_3)
        cor_3_2n = 0

        for i in range(300):
            if ind_3nn_2[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_3nn_2[i][1]]
            elif ind_3nn_2[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_3nn_2[i][0]]   
            elif ind_3nn_2[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_3nn_2[i][1]]
            elif ind_3nn_2[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_3nn_2[i][0]]
            else:             
                val1 = ele_list_s[ind_3nn_2[i][0]]
                val2 = ele_list_s[ind_3nn_2[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_3_2n += cor_func
            cor_func_new = abs(cor_3_2n - ideal_3)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

@cuda.jit
def list_action_4(a1_value, ele_list_s, cor_4):

    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        # ?cor_4
        b_ind = cuda.blockIdx.x #*a1
        ideal_4 = 0

        cor_func_raw = abs(cor_4 - ideal_4)
        cor_4n = 0       
        for i in range(288):
            if ind_4nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_4nn[i][1]]
            elif ind_4nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_4nn[i][0]]   
            elif ind_4nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_4nn[i][1]]
            elif ind_4nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_4nn[i][0]]
            else:             
                val1 = ele_list_s[ind_4nn[i][0]]
                val2 = ele_list_s[ind_4nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_4n += cor_func

            cor_func_new = abs(cor_4n - ideal_4)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

def list_action(a1_value, ele_list_s,
                cor_1_1, cor_1_2, cor_2, cor_3_1, cor_3_2, cor_4):

    # idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    t_ind = cuda.threadIdx.x #*a2
    if t_ind < 108:
        # ?cor_1_1
        b_ind = cuda.blockIdx.x #*a1

        # ideal_1, ideal_2, ideal_3, ideal_4 = 0,0,0,0
        ideal_1 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_1_1 - ideal_1)
        # !shuffle array

        # len0, len1, len_1 = 0, 0, 0
        # if t_ind == 107:
        #     for i in range(len(ele_list_s)):
        #         if ele_list_s[i] == 0:
        #             len0 += 1
        #         if ele_list_s[i] == 1:
        #             len1 += 1
        #         if ele_list_s[i] == -1:
        #             len_1 += 1

        #     print(len0, len1, len_1)

        # !PROBLEM
        cor_1_1n = 0

        for i in range(225):
            if ind_1nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_1nn[i][1]]
            elif ind_1nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_1nn[i][0]]   
            elif ind_1nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_1nn[i][1]]
            elif ind_1nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_1nn[i][0]]
            else:             
                val1 = ele_list_s[ind_1nn[i][0]]
                val2 = ele_list_s[ind_1nn[i][1]]
            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_1_1n += cor_func
            cor_func_new = abs(cor_1_1n - ideal_1)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind] += swap_reward

    elif 108 <= t_ind < 216:
        # ?cor_1_2
        b_ind = cuda.blockIdx.x #*a1
        ideal_1 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_1_2 - ideal_1)
        cor_1_2n = 0

        for i in range(225, 450):
            if ind_1nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_1nn[i][1]]
            elif ind_1nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_1nn[i][0]]   
            elif ind_1nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_1nn[i][1]]
            elif ind_1nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_1nn[i][0]]
            else:             
                val1 = ele_list_s[ind_1nn[i][0]]
                val2 = ele_list_s[ind_1nn[i][1]]
            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_1_2n += cor_func
            cor_func_new = abs(cor_1_2n - ideal_1)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind-108] += swap_reward            

    elif 216 <= t_ind < 324:
        # ?cor_2
        b_ind = cuda.blockIdx.x #*a1
        ideal_2 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_2 - ideal_2)
        cor_2n = 0

        for i in range(216):
            if ind_2nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_2nn[i][1]]
            elif ind_2nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_2nn[i][0]]   
            elif ind_2nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_2nn[i][1]]
            elif ind_2nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_2nn[i][0]]
            else:             
                val1 = ele_list_s[ind_2nn[i][0]]
                val2 = ele_list_s[ind_2nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_2n += cor_func
            cor_func_new = abs(cor_2n - ideal_2)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind-216] += swap_reward

    elif 324 <= t_ind < 432:
        # ?cor_3_1
        b_ind = cuda.blockIdx.x #*a1
        ideal_3 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_3_1 - ideal_3)
        cor_3_1n = 0

        for i in range(300):
            if ind_3nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_3nn[i][1]]
            elif ind_3nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_3nn[i][0]]   
            elif ind_3nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_3nn[i][1]]
            elif ind_3nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_3nn[i][0]]
            else:             
                val1 = ele_list_s[ind_3nn[i][0]]
                val2 = ele_list_s[ind_3nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_3_1n += cor_func
            cor_func_new = abs(cor_3_1n - ideal_3)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind-324] += swap_reward

    elif 432 <= t_ind < 540:
        # ?cor_3_2
        b_ind = cuda.blockIdx.x #*a1
        ideal_3 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_3_2 - ideal_3)
        cor_3_2n = 0

        for i in range(300, 600):
            if ind_3nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_3nn[i][1]]
            elif ind_3nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_3nn[i][0]]   
            elif ind_3nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_3nn[i][1]]
            elif ind_3nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_3nn[i][0]]
            else:             
                val1 = ele_list_s[ind_3nn[i][0]]
                val2 = ele_list_s[ind_3nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_3_2n += cor_func
            cor_func_new = abs(cor_3_2n - ideal_3)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind-432] += swap_reward

    elif 540 <= t_ind < 648:
        # ?cor_4
        b_ind = cuda.blockIdx.x #*a1
        ideal_4 = 0

        # !swap step cor func calc
        cor_func_raw = abs(cor_4 - ideal_4)
        cor_4n = 0       
        for i in range(288):
            if ind_4nn[i][0] == t_ind:
                val1 = ele_list_s[b_ind]
                val2 = ele_list_s[ind_4nn[i][1]]
            elif ind_4nn[i][1] == t_ind:
                val2 = ele_list_s[b_ind]
                val1 = ele_list_s[ind_4nn[i][0]]   
            elif ind_4nn[i][0] == b_ind:
                val1 = ele_list_s[t_ind]
                val2 = ele_list_s[ind_4nn[i][1]]
            elif ind_4nn[i][1] == b_ind:
                val2 = ele_list_s[t_ind]
                val1 = ele_list_s[ind_4nn[i][0]]
            else:             
                val1 = ele_list_s[ind_4nn[i][0]]
                val2 = ele_list_s[ind_4nn[i][1]]

            phi1_1 = math.sqrt(3/2)*val1
            phi2_1 = math.sqrt(3/2)*val2
            phi1_2 = math.sqrt(2)*(3/2*(val1**2) - 1)
            phi2_2 = math.sqrt(2)*(3/2*(val2**2) - 1)
            cor_func = phi1_1*phi2_1+phi1_1*phi2_2+phi1_2*phi2_1+phi1_2*phi2_2

            cor_4n += cor_func

            cor_func_new = abs(cor_4n - ideal_4)
            swap_reward = cor_func_new - cor_func_raw

            a1_value[b_ind, t_ind-540] += swap_reward

def main(iter_time):
    block = 108
    thread = 128

    ele_list, a1_list = [], []
    for _ in range(iter_time):
        # ?rawlist
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
        list_action_1_1[block, thread](a1_value, raw_list, cor_1_1)
        cuda.synchronize()
        list_action_1_2[block, thread](a1_value, raw_list, cor_1_2)
        cuda.synchronize()
        list_action_2[block, thread](a1_value, raw_list, cor_2)
        cuda.synchronize()
        list_action_3_1[block, thread](a1_value, raw_list, cor_3_1)
        cuda.synchronize()
        list_action_3_2[block, thread](a1_value, raw_list, cor_3_2)
        cuda.synchronize()
        list_action_4[block, thread](a1_value, raw_list, cor_4)
        cuda.synchronize()
        
        # !print(raw_list)
        a1_value = np.sum(a1_value.copy_to_host(), axis=1)
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
    np.save('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/cuda_a1list.npy', a1_list)
    np.save('/media/wz/7AD631A4D6316195/Projects/SQS_drl/fcc_108/cuda_elelist.npy', ele_list)

    second_to_hour(time.time() - start_)
