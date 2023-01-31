import numpy as np
from sub_func import abs_dis, ele_list_gen, cor_func_all, swap_step, atom_get
from random import randrange
import argparse

import math
import multiprocessing as mp
import time

cr_, mn_, co_, ni_ = atom_get()
target_cor = 4000
benchmark_test = True

# print(f'cr, mn, co, ni content: {cr_, mn_, co_, ni_}')

def remove_none(a):
    return a[a != None]

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

def main(iter):
    start_ = time.time()
    ele_list = ele_list_gen(cr_, mn_, co_, ni_, mode='randchoice')
    cor_func = cor_func_all(ele_list)

    step_count = 0
    cor_func_raw = cor_func
    while True:
        action = np.array([randrange(192), randrange(192)])
        ele_list_n, r, cor_func_n, _ = swap_step(action, cor_func_raw, ele_list, 0)
        r_ = np.exp(r/3)
        if np.random.rand() <= np.min([r_, 1]) and cor_func_n != cor_func_raw:
            ele_list = ele_list_n
            cor_func_raw = cor_func_n
            step_count += 1

        if cor_func_n < target_cor:
            print(f'iter: {iter}, step: {step_count}')
            break
    if benchmark_test:
        return [time.time() - start_, step_count]
    else:
        return ele_list_n.tolist()
    
            
def multicore(iter_time, process_num):
    # pool = mp.Pool(processes=2)
    pool = mp.Pool(processes=process_num)
    output_list = [pool.map(main, range(iter_time))]
    # map equation to the value
    return output_list

if __name__ == '__main__':
    print(cr_, mn_, co_, ni_)
    iter_time = 100

    start_ = time.time()
    output_list = [multicore(iter_time, process_num=4)][0][0]
    output_list = [i for i in output_list if i]
    np.save(f'./Cr{int(cr_*100)}Mn{int(mn_*100)}Co{int(co_*100)}_{target_cor}_try2_pbc.npy', output_list)
    second_to_hour(time.time() - start_)
    try:
        print(len(output_list))
    except:
        print('Sadly nothing reached the goal')
