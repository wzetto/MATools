import concurrent.futures
import numpy as np
import para_orstruc_func
import math
import time

cr_content, co_content = 0.45, 0.05
ideal_fir = para_orstruc_func.ideal_cor_func(cr_content, co_content, 935)
ideal_sec = para_orstruc_func.ideal_cor_func(cr_content, co_content, 387)
ideal_thr = para_orstruc_func.ideal_cor_func(cr_content, co_content, 1394)
ideal_for = para_orstruc_func.ideal_cor_func(cr_content, co_content, 621)

ideal_fir_inter2345 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 593)
ideal_sec_inter2345 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 232)
ideal_thr_inter2345 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 824)
ideal_for_inter2345 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 363)

ideal_fir_inter34 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 251)
ideal_sec_inter34 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 77)
ideal_thr_inter34 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 256)
ideal_for_inter34 = para_orstruc_func.ideal_cor_func(cr_content, co_content, 104)

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

def Mutation_raw(input_array, mode):#return the mutation list
    
    len_cr = len(input_array[0])
    len_co = len(input_array[1])
    len_ni = len(input_array[2])
    len_s = len(input_array[3])
    ind_raw = input_array[3]
    
    index_l1 = np.where(np.array(ind_raw)[:,2] == 0)[0]
    index_l2 = np.where(np.array(ind_raw)[:,2] == math.sqrt(1/3))[0]
    index_l3 = np.where(np.array(ind_raw)[:,2] == math.sqrt(1/3)*2)[0]
    index_l4 = np.where(np.array(ind_raw)[:,2] == math.sqrt(1/3)*3)[0]
    index_l5 = np.where(np.array(ind_raw)[:,2] == math.sqrt(1/3)*4)[0]
    index_l6 = np.where(np.array(ind_raw)[:,2] == math.sqrt(1/3)*5)[0]
    index_l34 = np.concatenate([index_l3, index_l4])#index of 3,4 layers
    index_l25 = np.concatenate([index_l2, index_l5])#index of 2,5 layers
    index_l16 = np.concatenate([index_l1, index_l6])#index of 1,6 layers
    
    epsilon_ = np.random.rand()
    if epsilon_ < 0.6:
        repeat_time = np.random.randint(2,8)
    elif epsilon_ >= 0.6:
        repeat_time = np.random.randint(6,12)

    if mode=='Stage1':
        for time_ in range(repeat_time):
            point_3_ind = index_l34[np.random.randint(72)]
            point_4_ind = index_l34[np.random.randint(72)]
            point_3 = ind_raw[point_3_ind].copy()
            point_4 = ind_raw[point_4_ind].copy()
            ind_raw[point_3_ind] = point_4
            ind_raw[point_4_ind] = point_3
    
    elif mode=='Stage2':
        for time_ in range(repeat_time):
            point_5_ind = index_l25[np.random.randint(72)]
            point_6_ind = index_l25[np.random.randint(72)]
            point_5 = ind_raw[point_5_ind].copy()
            point_6 = ind_raw[point_6_ind].copy()
            ind_raw[point_5_ind] = point_6
            ind_raw[point_6_ind] = point_5
            
    elif mode=='Stage3':
        for time_ in range(repeat_time):
            point_5_ind = index_l16[np.random.randint(72)]
            point_6_ind = index_l16[np.random.randint(72)]
            point_5 = ind_raw[point_5_ind].copy()
            point_6 = ind_raw[point_6_ind].copy()
            ind_raw[point_5_ind] = point_6
            ind_raw[point_6_ind] = point_5
    
    elif mode=='Whole':
        for time_ in range(repeat_time):
            point_7_ind = np.random.randint(len_s)
            point_8_ind = np.random.randint(len_s)
            point_7 = ind_raw[point_7_ind].copy()
            point_8 = ind_raw[point_8_ind].copy()
            ind_raw[point_7_ind] = point_8
            ind_raw[point_8_ind] = point_7
            
    cr_pos_mut = ind_raw[0:len_cr]
    co_pos_mut = ind_raw[len_cr:len_cr+len_co]
    ni_pos_mut = ind_raw[len_cr+len_co:len_cr+len_co+len_ni]
    mut_gen = [cr_pos_mut, co_pos_mut, ni_pos_mut, ind_raw]
    return mut_gen

def mut_whole(input_array):
    mut_gen = Mutation_raw(input_array, mode='Whole')
    return mut_gen

def mut_34(input_array):
    mut_gen = Mutation_raw(input_array, mode='Stage1')
    return mut_gen

def mut_25(input_array):
    mut_gen = Mutation_raw(input_array, mode='Stage2')
    return mut_gen

def mut_16(input_array):
    mut_gen = Mutation_raw(input_array, mode='Stage3')
    return mut_gen

def Fitness(input_array):
    ##consider more bro?
    cr_pos = np.array(input_array[0])
    co_pos = np.array(input_array[1])
    ni_pos = np.array(input_array[2])
    ind_raw = np.array(input_array[3])

    cor_func_val_1 = para_orstruc_func.nn_cor_func_new(cr_pos, co_pos, ni_pos, ind_raw, 0.707107, threshold=0.01)
    cor_func_val_2 = para_orstruc_func.nn_cor_func_new(cr_pos, co_pos, ni_pos, ind_raw, 1, threshold=0.01)
    cor_func_val_3 = para_orstruc_func.nn_cor_func_new(cr_pos, co_pos, ni_pos, ind_raw, 1.224745, threshold=0.01)
    cor_func_val_4 = para_orstruc_func.nn_cor_func_new(cr_pos, co_pos, ni_pos, ind_raw, 1.414214, threshold=0.01)

    res_1 = abs(ideal_fir - cor_func_val_1)
    res_2 = abs(ideal_sec - cor_func_val_2)
    res_3 = abs(ideal_thr - cor_func_val_3)
    res_4 = abs(ideal_for - cor_func_val_4)
    res_1_ratio = res_1/ideal_fir
    res_2_ratio = res_2/ideal_sec
    res_3_ratio = res_3/ideal_thr
    res_4_ratio = res_4/ideal_for

    cor_func_inter2345val_1 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 0.707107, [0.5, 2.5], threshold=0.01)
    cor_func_inter2345val_2 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 1, [0.5, 2.5], threshold=0.01)
    cor_func_inter2345val_3 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 1.224745, [0.5, 2.5], threshold=0.01)
    cor_func_inter2345val_4 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 1.414214, [0.5, 2.5], threshold=0.01)

    res_1_in2345 = abs(ideal_fir_inter2345 - cor_func_inter2345val_1)
    res_2_in2345 = abs(ideal_sec_inter2345 - cor_func_inter2345val_2)
    res_3_in2345 = abs(ideal_thr_inter2345 - cor_func_inter2345val_3)
    res_4_in2345 = abs(ideal_for_inter2345 - cor_func_inter2345val_4)
    res_1_in2345_ratio = res_1_in2345/ideal_fir_inter2345
    res_2_in2345_ratio = res_2_in2345/ideal_sec_inter2345
    res_3_in2345_ratio = res_3_in2345/ideal_thr_inter2345
    res_4_in2345_ratio = res_4_in2345/ideal_for_inter2345

    cor_func_inter34val_1 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 0.707107, [1, 2], threshold=0.01)
    cor_func_inter34val_2 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 1, [1, 2], threshold=0.01)
    cor_func_inter34val_3 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 1.224745, [1, 2], threshold=0.01)
    cor_func_inter34val_4 = para_orstruc_func.nn_cor_func_inter(cr_pos, co_pos, ni_pos, ind_raw, 1.414214, [1, 2], threshold=0.01)
    
    res_1_in34 = abs(ideal_fir_inter34 - cor_func_inter34val_1)
    res_2_in34 = abs(ideal_sec_inter34 - cor_func_inter34val_2)
    res_3_in34 = abs(ideal_thr_inter34 - cor_func_inter34val_3)
    res_4_in34 = abs(ideal_for_inter34 - cor_func_inter34val_4)
    res_1_in34_ratio = res_1_in34/ideal_fir_inter34
    res_2_in34_ratio = res_2_in34/ideal_sec_inter34
    res_3_in34_ratio = res_3_in34/ideal_thr_inter34
    res_4_in34_ratio = res_4_in34/ideal_for_inter34

    p_whole = 1
    p_2345 = 2
    p_34 = 5

    fitness = np.linalg.norm([res_1*p_whole,res_2*p_whole,res_3*p_whole,res_4*p_whole,
                            res_1_in2345*p_2345,res_2_in2345*p_2345,
                            res_3_in2345*p_2345,res_4_in2345*p_2345,
                            res_1_in34*p_34,res_2_in34*p_34,res_3_in34*p_34,res_4_in34*p_34])

    fitness_34 = np.linalg.norm([res_1_in34, res_2_in34, res_3_in34, res_4_in34])

    fitness_ratio = np.array([res_1_ratio,res_2_ratio,res_3_ratio,res_4_ratio,
                            res_1_in2345_ratio,res_2_in2345_ratio,res_3_in2345_ratio,res_4_in2345_ratio,
                            res_1_in34_ratio,res_2_in34_ratio,res_3_in34_ratio,res_4_in34_ratio])
        
    fitness_part_ratio = np.array([np.mean(fitness_ratio[0:4]),np.mean(fitness_ratio[4:8]),np.mean(fitness_ratio[8:12])])
    fitness_ratio_34 = np.array([res_1_in34_ratio,res_2_in34_ratio,res_3_in34_ratio,res_4_in34_ratio])

    return fitness, fitness_34, np.mean(fitness_ratio), fitness_part_ratio, fitness_ratio_34

def chosen_ind_list(fitness_whole, repeat_time):

    prob_list = np.array([(1/(fitness_whole[i3]/np.sum(fitness_whole))) for i3 in range(len(fitness_whole))])
    prob_list = (1-prob_list)/np.sum(1-prob_list)
    prob_list_neo = np.zeros(len(prob_list))
    for i4 in range(len(prob_list)):
        prob_list_neo[i4] = np.sum(prob_list[:i4+1])

    chosen_ind = []
    for _ in range(repeat_time):
        god_dice = np.random.rand()
        prob_sign = np.sign(god_dice - prob_list_neo)
        god_chosen_ind = np.where(prob_sign == -1)[0][0]#the chosen index
        chosen_ind.append(god_chosen_ind)

    return chosen_ind

def lexsort(input_list):
    cr_list = np.array(input_list[0])
    cr_list = cr_list[np.lexsort((
        cr_list[:,2], cr_list[:,1], cr_list[:,0]))]

    co_list = np.array(input_list[1])
    co_list = co_list[np.lexsort((
        co_list[:,2], co_list[:,1], co_list[:,0]))]

    ni_list = np.array(input_list[2])
    ni_list = ni_list[np.lexsort((
        ni_list[:,2], ni_list[:,1], ni_list[:,0]))]

    return np.concatenate([cr_list, co_list, ni_list], axis=0).tolist()

def kill_dup(input_gen):#kill the dup of complicate list
    origin_mut_whole, origin_mut_test = [], []
    for origin_whole in input_gen:
        if len(origin_mut_whole) == 0:
            origin_mut_whole.append(origin_whole)
            origin_mut_test.append(lexsort(origin_whole))
        else:
            origin_mut_testnorm = np.array(origin_mut_test)-np.array(lexsort(origin_whole))
            origin_mut_testnorm = np.array([np.linalg.norm(i) for i in origin_mut_testnorm])
            if len(origin_mut_testnorm[origin_mut_testnorm == 0]) == 0:
                origin_mut_whole.append(origin_whole)
                origin_mut_test.append(lexsort(origin_whole))
    
    return origin_mut_whole

origin_list = np.load(
    f'C:/Users/yaoho/OneDrive - Kyoto University/Project/SFE/origin_list{int(cr_content*100)}{int(co_content*100)}.npy',
    allow_pickle=True).tolist()
origin_list = origin_list[:7]
fitness_whole = []
fitness_34 = []
fitness_mean = []

if __name__ == '__main__':
    #origin input
    start_ = time.time()

    for i in range(7):
        fitness_whole.append(Fitness(origin_list[i])[0])
        fitness_34.append(Fitness(origin_list[i])[1])

    try_step = 5000
    for kkt in range(try_step):

        #seperate to each origin_list, now 24
        epsilon = np.random.random()
        chosen_ind = chosen_ind_list(
            epsilon*np.array(fitness_whole)+(1-epsilon)*np.array(fitness_34), 24)

        origin_list_chosen = [origin_list[c_ind] for c_ind in chosen_ind]

        with concurrent.futures.ProcessPoolExecutor() as executor:

            origin_mut_whole_gen = executor.map(mut_whole, origin_list_chosen)
            if kkt < int(try_step/4):
                origin_mut_layer_gen = executor.map(mut_34, origin_list_chosen)
            elif int(try_step/4) <= kkt < int(5/8*try_step):
                origin_mut_layer_gen = executor.map(mut_25, origin_list_chosen)
            elif int(5/8*try_step) <= kkt:
                origin_mut_layer_gen = executor.map(mut_16, origin_list_chosen)

            #除重
            origin_mut_whole = kill_dup(origin_mut_whole_gen)
            origin_mut_layer = kill_dup(origin_mut_layer_gen)

            fitness_res_whole_gen = executor.map(Fitness, origin_mut_whole)
            fitness_res_layer_gen = executor.map(Fitness, origin_mut_layer) 
            # to calc the fitness of each origin_list
            fitness_res_whole, fitness_res_layer = [], []

            for fit_res_layer in fitness_res_layer_gen:
                fitness_res_layer.append(fit_res_layer[:2])

            for fit_res_whole in fitness_res_whole_gen:
                fitness_res_whole.append(fit_res_whole[:2])

            # print(fitness_res_whole)

            for i in range(len(fitness_res_whole)):
                if (fitness_res_whole[i][0] < np.mean(fitness_whole)
                    and fitness_res_whole[i][1] < np.mean(fitness_34)):
                    origin_list.append(origin_mut_whole[i])
                    fitness_whole.append(fitness_res_whole[i][0])
                    fitness_34.append(fitness_res_whole[i][1])

            for i in range(len(fitness_res_layer)):
                if (fitness_res_layer[i][0] < np.mean(fitness_whole)
                    and fitness_res_layer[i][1] < np.mean(fitness_34)):
                    origin_list.append(origin_mut_layer[i])
                    fitness_whole.append(fitness_res_layer[i][0])
                    fitness_34.append(fitness_res_layer[i][1])

            if kkt%50 == 0 or kkt==try_step-1:
                origin_list = kill_dup(origin_list)

                fitness_whole = []
                fitness_34 = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    fitness_ratio_gen = executor.map(Fitness, origin_list)

                    for fit_ratio in fitness_ratio_gen:
                        fitness_whole.append(fit_ratio[0])
                        fitness_34.append(fit_ratio[1])
    
        if len(origin_list) >= 400:
            new_type = []
            new_type_ind = np.where(fitness_whole <= np.mean(fitness_whole))[0]
            if len(new_type_ind) >= 100:
                for nt in new_type_ind:
                    new_type.append(origin_list[nt])
                new_type_red = kill_dup(new_type)
                if len(new_type_red) >= 60:
                    origin_list = new_type_red.copy()

                    fitness_whole = []
                    fitness_34 = []
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        fitness_ratio_gen = executor.map(Fitness, origin_list)

                        for fit_ratio in fitness_ratio_gen:
                            fitness_whole.append(fit_ratio[0])
                            fitness_34.append(fit_ratio[1])

        assert len(origin_list) == len(fitness_whole), f'Not same length {len(origin_list)} {len(fitness_whole)}'

        if kkt%20 == 0:
            print(f'Fitness mean: {np.mean(fitness_whole)} iter {kkt} length of list {len(origin_list)}')
            fitness_mean.append(np.mean(fitness_whole))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        fitness_ratio_gen = executor.map(Fitness, origin_list)

        fitness_ratio, fitness_ = [], []
        for fit_ratio in fitness_ratio_gen:
            fitness_ratio.append(fit_ratio[2])
            fitness_.append(fit_ratio[0])

    np.save(f'C:/Users/yaoho/OneDrive - Kyoto University/Project/SFE/origin_list/origin_list{int(cr_content*100)}{int(co_content*100)}.npy', origin_list)
    np.save(f'C:/Users/yaoho/OneDrive - Kyoto University/Project/SFE/origin_list/fitness_result{int(cr_content*100)}{int(co_content*100)}.npy', fitness_)
    np.save(f'C:/Users/yaoho/OneDrive - Kyoto University/Project/SFE/origin_list/fitness_whole{int(cr_content*100)}{int(co_content*100)}.npy', fitness_mean)
    np.save(f'C:/Users/yaoho/OneDrive - Kyoto University/Project/SFE/origin_list/fitness_ratio{int(cr_content*100)}{int(co_content*100)}.npy', fitness_ratio)
      
    second_to_hour(time.time() - start_)
