import multiprocessing as mp
import time
import numpy as np
import math
import sub_func_mc
from apscheduler.schedulers.blocking import BlockingScheduler

cr_content, co_content = 0.45, 0.15

ideal_fir = sub_func_mc.ideal_cor_func(cr_content, co_content, 935)
ideal_sec = sub_func_mc.ideal_cor_func(cr_content, co_content, 387)
ideal_thr = sub_func_mc.ideal_cor_func(cr_content, co_content, 1394)
ideal_for = sub_func_mc.ideal_cor_func(cr_content, co_content, 621)

ideal_fir_inter2345 = sub_func_mc.ideal_cor_func(cr_content, co_content, 593)
ideal_sec_inter2345 = sub_func_mc.ideal_cor_func(cr_content, co_content, 232)
ideal_thr_inter2345 = sub_func_mc.ideal_cor_func(cr_content, co_content, 824)
ideal_for_inter2345 = sub_func_mc.ideal_cor_func(cr_content, co_content, 363)

ideal_fir_inter34 = sub_func_mc.ideal_cor_func(cr_content, co_content, 251)
ideal_sec_inter34 = sub_func_mc.ideal_cor_func(cr_content, co_content, 77)
ideal_thr_inter34 = sub_func_mc.ideal_cor_func(cr_content, co_content, 256)
ideal_for_inter34 = sub_func_mc.ideal_cor_func(cr_content, co_content, 104)

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

def single_gen(iter):

    norm_stack_simp = sub_func_mc.simp_struc_gen(mode = 'simp')
    cr_pos, co_pos, ni_pos, ind_raw = sub_func_mc.gen_crconi_sfe(
        cr_content, co_content, norm_stack_simp)
    #whole sys
    cor_func_val_1 = sub_func_mc.nn_cor_func_new(
        cr_pos, co_pos, ni_pos, ind_raw, 0.707107, threshold=0.01)
    cor_func_val_2 = sub_func_mc.nn_cor_func_new(
        cr_pos, co_pos, ni_pos, ind_raw, 1, threshold=0.01)
    cor_func_val_3 = sub_func_mc.nn_cor_func_new(
        cr_pos, co_pos, ni_pos, ind_raw, 1.224745, threshold=0.01)
    cor_func_val_4 = sub_func_mc.nn_cor_func_new(
        cr_pos, co_pos, ni_pos, ind_raw, 1.414214, threshold=0.01)

    #layer 2,3,4,5
    cor_func_inter2345val_1 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 0.707107, [0.5, 2.5], threshold=0.01)
    cor_func_inter2345val_2 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 1, [0.5, 2.5], threshold=0.01)
    cor_func_inter2345val_3 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 1.224745, [0.5, 2.5], threshold=0.01)
    cor_func_inter2345val_4 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 1.414214, [0.5, 2.5], threshold=0.01)

    #layer 3,4
    cor_func_inter34val_1 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 0.707107, [1, 2], threshold=0.01)
    cor_func_inter34val_2 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 1, [1, 2], threshold=0.01)
    cor_func_inter34val_3 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 1.224745, [1, 2], threshold=0.01)
    cor_func_inter34val_4 = sub_func_mc.nn_cor_func_inter(
        cr_pos, co_pos, ni_pos, ind_raw, 1.414214, [1, 2], threshold=0.01)
    
    distri_raw = np.array([
        cor_func_val_1/ideal_fir - 1,
        cor_func_val_2/ideal_sec - 1,
        cor_func_val_3/ideal_thr - 1,
        cor_func_val_4/ideal_for - 1,
        cor_func_inter2345val_1/ideal_fir_inter2345 - 1,
        cor_func_inter2345val_2/ideal_sec_inter2345 - 1,
        cor_func_inter2345val_3/ideal_thr_inter2345 - 1,
        cor_func_inter2345val_4/ideal_for_inter2345 - 1,
        cor_func_inter34val_1/ideal_fir_inter34 - 1,
        cor_func_inter34val_2/ideal_sec_inter34 - 1,
        cor_func_inter34val_3/ideal_thr_inter34 - 1,
        cor_func_inter34val_4/ideal_for_inter34 - 1,
    ])
    if iter%50000 == 0:
        print(iter)
    if np.max(np.abs(distri_raw)) < 0.08:
        print(f'one done {iter}')
        return [cr_pos.tolist(), co_pos.tolist(), ni_pos.tolist()]
    else:
        pass

def multicore(iter_time, process_num):
    # pool = mp.Pool(processes=2)#
    pool = mp.Pool(processes=process_num)
    output_list = [pool.map(single_gen, range(iter_time))]
    # map equation to the value
    return output_list


def main():
    
    start_ = time.time()
    output_list = [multicore(iter_time=500000, process_num=12)][0][0]
    output_list = [i for i in output_list if i]
    np.save(f'rawlist_cr{int(cr_content*100)}co{int(co_content*100)}.npy', output_list)
    second_to_hour(time.time() - start_)
    try:
        print(len(output_list))
    except:
        print('sadly nothing reached goal')

if __name__ == '__main__':
    sched = BlockingScheduler()
    sched.add_job(main, 'cron', hour=13, minute=18)
    sched.start()


