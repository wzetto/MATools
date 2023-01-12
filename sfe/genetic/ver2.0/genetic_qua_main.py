import numpy as np
import sub_func
from sub_func import abs_dis, ideal_cor, cor_func, ele_list_gen
import math
import time
from itertools import combinations
from random import randrange
import concurrent.futures

ind_1nn = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_1nn.npy')
ind_2nn = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_2nn.npy')
ind_3nn = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_3nn.npy')
ind_4nn = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_4nn.npy')

ind_1nn_345 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_1nn_345.npy')
ind_2nn_345 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_2nn_345.npy')
ind_3nn_345 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_3nn_345.npy')
ind_4nn_345 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_4nn_345.npy')

ind_1nn_34 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_1nn_34.npy')
ind_2nn_34 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_2nn_34.npy')
ind_3nn_34 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_3nn_34.npy')
ind_4nn_34 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_4nn_34.npy')

ind_1nn_45 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_1nn_45.npy')
ind_2nn_45 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_2nn_45.npy')
ind_3nn_45 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_3nn_45.npy')
ind_4nn_45 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_4nn_45.npy')

ind_layer_345 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_l_345.npy')
ind_layer_34 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_l_34.npy')
ind_layer_45 = np.load('/media/wz/7AD631A4D6316195/Projects/GSFE/fcc_216/ind_l_45.npy')

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

#######################################
class genetic:
    def __init__(self, cr_, mn_, co_, ni_,
                ind_1nn, ind_2nn, ind_3nn, ind_4nn,
                ind_1nn_345, ind_2nn_345, ind_3nn_345, ind_4nn_345,
                ind_1nn_34, ind_2nn_34, ind_3nn_34, ind_4nn_34,
                ind_1nn_45, ind_2nn_45, ind_3nn_45, ind_4nn_45,
                ind_layer_345, ind_layer_34, ind_layer_45,
                state):
        
        assert abs(cr_ + mn_ + co_ + ni_ - 1) < 0.001, 'Make sure atomic ratio summing to 1'
        self.ind_1nn, self.ind_2nn, self.ind_3nn, self.ind_4nn = ind_1nn, ind_2nn, ind_3nn, ind_4nn
        self.ind_1nn_345, self.ind_2nn_345, self.ind_3nn_345, self.ind_4nn_345 = ind_1nn_345, ind_2nn_345, ind_3nn_345, ind_4nn_345
        self.ind_1nn_34, self.ind_2nn_34, self.ind_3nn_34, self.ind_4nn_34 = ind_1nn_34, ind_2nn_34, ind_3nn_34, ind_4nn_34
        self.ind_1nn_45, self.ind_2nn_45, self.ind_3nn_45, self.ind_4nn_45 = ind_1nn_45, ind_2nn_45, ind_3nn_45, ind_4nn_45

        self.ideal_1 = ideal_cor(cr_, mn_, co_, ind_1nn)
        self.ideal_2 = ideal_cor(cr_, mn_, co_, ind_2nn)
        self.ideal_3 = ideal_cor(cr_, mn_, co_, ind_3nn)
        self.ideal_4 = ideal_cor(cr_, mn_, co_, ind_4nn)

        self.ideal_1_345 = ideal_cor(cr_, mn_, co_, ind_1nn_345)
        self.ideal_2_345 = ideal_cor(cr_, mn_, co_, ind_2nn_345)
        self.ideal_3_345 = ideal_cor(cr_, mn_, co_, ind_3nn_345)
        self.ideal_4_345 = ideal_cor(cr_, mn_, co_, ind_4nn_345)

        self.ideal_1_34 = ideal_cor(cr_, mn_, co_, ind_1nn_34)
        self.ideal_2_34 = ideal_cor(cr_, mn_, co_, ind_2nn_34)
        self.ideal_3_34 = ideal_cor(cr_, mn_, co_, ind_3nn_34)
        self.ideal_4_34 = ideal_cor(cr_, mn_, co_, ind_4nn_34)

        self.ideal_1_45 = ideal_cor(cr_, mn_, co_, ind_1nn_45)
        self.ideal_2_45 = ideal_cor(cr_, mn_, co_, ind_2nn_45)
        self.ideal_3_45 = ideal_cor(cr_, mn_, co_, ind_3nn_45)
        self.ideal_4_45 = ideal_cor(cr_, mn_, co_, ind_4nn_45)

        self.ind_345, self.ind_34, self.ind_45 = ind_layer_345, ind_layer_34, ind_layer_45
        self.state = state

    def swap_step(self, state, action):
        a1, a2 = action
        state[a1], state[a2] = state[a2], state[a1]
        return state

    def mutation_raw(self, mode, state):
        epsilon_ = np.random.rand()
        if epsilon_ < 0.6:
            repeat_time = np.random.randint(2, 8)
        elif epsilon_ >= 0.6:
            repeat_time = np.random.randint(6, 12)

        if mode == 'whole':
            for time_ in range(repeat_time):
                action = np.array([randrange(216), randrange(216)])
                state = self.swap_step(state, action)

        elif mode == 'duo34':
            for time_ in range(repeat_time):
                action = np.array([np.random.choice(self.ind_34), np.random.choice(self.ind_34)])
                state = self.swap_step(state, action)
        
        elif mode == 'duo45':
            for time_ in range(repeat_time):
                action = np.array([np.random.choice(self.ind_45), np.random.choice(self.ind_45)])
                state = self.swap_step(state, action)

        elif mode == 'tri':
            for time_ in range(repeat_time):
                action = np.array([np.random.choice(self.ind_345), np.random.choice(self.ind_345)])
                state = self.swap_step(state, action)

        return state
            
    def mut_whole(self, state):
        return self.mutation_raw('whole', state)

    def mut_34(self, state):
        return self.mutation_raw('duo34', state)

    def mut_45(self, state):
        return self.mutation_raw('duo45', state)

    def mut_345(self, state):
        return self.mutation_raw('tri', state)

    def Fitness(self, state):
        
        cor_func_1 = cor_func(self.ind_1nn, state,)
        cor_func_2 = cor_func(self.ind_2nn, state,)
        cor_func_3 = cor_func(self.ind_3nn, state,)
        cor_func_4 = cor_func(self.ind_4nn, state,)

        res_1 = np.abs(cor_func_1 - self.ideal_1)
        res_2 = np.abs(cor_func_2 - self.ideal_2)
        res_3 = np.abs(cor_func_3 - self.ideal_3)
        res_4 = np.abs(cor_func_4 - self.ideal_4)

        res_whole = np.concatenate([res_1, res_2, res_3, res_4])

        cor_func_1_345 = cor_func(self.ind_1nn_345, state,)
        cor_func_2_345 = cor_func(self.ind_2nn_345, state,)
        cor_func_3_345 = cor_func(self.ind_3nn_345, state,)
        cor_func_4_345 = cor_func(self.ind_4nn_345, state,)

        res_1_345 = np.abs(cor_func_1_345 - self.ideal_1_345)
        res_2_345 = np.abs(cor_func_2_345 - self.ideal_2_345)
        res_3_345 = np.abs(cor_func_3_345 - self.ideal_3_345)
        res_4_345 = np.abs(cor_func_4_345 - self.ideal_4_345)

        res_345 = np.concatenate([res_1_345, res_2_345, res_3_345, res_4_345])

        cor_func_1_34 = cor_func(self.ind_1nn_34, state,)
        cor_func_2_34 = cor_func(self.ind_2nn_34, state,)
        cor_func_3_34 = cor_func(self.ind_3nn_34, state,)
        cor_func_4_34 = cor_func(self.ind_4nn_34, state,)

        res_1_34 = np.abs(cor_func_1_34 - self.ideal_1_34)
        res_2_34 = np.abs(cor_func_2_34 - self.ideal_2_34)
        res_3_34 = np.abs(cor_func_3_34 - self.ideal_3_34)
        res_4_34 = np.abs(cor_func_4_34 - self.ideal_4_34)

        res_34 = np.concatenate([res_1_34, res_2_34, res_3_34, res_4_34])

        cor_func_1_45 = cor_func(self.ind_1nn_45, state,)
        cor_func_2_45 = cor_func(self.ind_2nn_45, state,)
        cor_func_3_45 = cor_func(self.ind_3nn_45, state,)
        cor_func_4_45 = cor_func(self.ind_4nn_45, state,)

        res_1_45 = np.abs(cor_func_1_45 - self.ideal_1_45)
        res_2_45 = np.abs(cor_func_2_45 - self.ideal_2_45)
        res_3_45 = np.abs(cor_func_3_45 - self.ideal_3_45)
        res_4_45 = np.abs(cor_func_4_45 - self.ideal_4_45)

        res_45 = np.concatenate([res_1_45, res_2_45, res_3_45, res_4_45])

        p_whole, p_345, p_34, p_45 = 2, 3, 3, 3
        res_overall = np.concatenate([res_whole*p_whole, res_345*p_345, res_34*p_34, res_45*p_45])
        fitness = np.linalg.norm(res_overall)
        fitness_345 = np.linalg.norm(res_345)
        fitness_34, fitness_45 = np.linalg.norm(res_34), np.linalg.norm(res_45)  

        return fitness, fitness_345, fitness_34, fitness_45    
    
    def chosen_ind_list(self, fitness, repeat_time):
        
        fitness = -fitness
        p_fit = np.exp(fitness) / np.sum(np.exp(fitness))
        chosen_ind = np.random.choice(range(len(fitness)), size = repeat_time, p=p_fit)

        return chosen_ind

    def gen_to_list(self, gene):
        origin_list = []
        for i in gene:
            origin_list.append(i.tolist())

        return np.array(origin_list)
    
if __name__ == '__main__':
    start_ = time.time()

    cr_, mn_, co_, ni_ = 1/4, 1/4, 1/4, 1/4
    origin_list, fitness_whole, fitness_345, fitness_mean = [], [], [], []

    for i in range(10):
        origin_list.append(ele_list_gen(cr_, mn_, co_, ni_).tolist())
    
    origin_list = np.load(f'/media/wz/7AD631A4D6316195/Projects/GSFE/origin_list_qua/origin_list{int(cr_*100)}{int(mn_*100)}{int(co_*100)}.npy')

    gen = genetic(cr_, mn_, co_, ni_,
                ind_1nn, ind_2nn, ind_3nn, ind_4nn,
                ind_1nn_345, ind_2nn_345, ind_3nn_345, ind_4nn_345,
                ind_1nn_34, ind_2nn_34, ind_3nn_34, ind_4nn_34,
                ind_1nn_45, ind_2nn_45, ind_3nn_45, ind_4nn_45,
                ind_layer_345, ind_layer_34, ind_layer_45,
                origin_list)

    for i in range(len(origin_list)):
        state = origin_list[i]
        fitness_whole.append(gen.Fitness(state)[0])
        fitness_345.append(gen.Fitness(state)[1])

    try_step = 30000
    max_work = 20
    for kkt in range(try_step):

        epsilon = np.random.random()
        fit_trans = epsilon*np.array(fitness_whole) + (1-epsilon)*np.array(fitness_345)

        origin_choseind = gen.chosen_ind_list(fit_trans, 24)
        origin_list_chosen = np.array([origin_list[i] for i in origin_choseind])

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_work) as executor:

            origin_mut_whole_gen = executor.map(gen.mut_whole, origin_list_chosen)
            origin_mut_whole_gen = gen.gen_to_list(origin_mut_whole_gen)
            if kkt < try_step//2:
                origin_mut_layer_gen = executor.map(gen.mut_345, origin_list_chosen)
                origin_mut_layer_gen = gen.gen_to_list(origin_mut_layer_gen)
            elif kkt >= try_step//2:
                origin_mut_layer_gen = executor.map(gen.mut_34, origin_list_chosen)
                origin_mut_layer_gen = gen.gen_to_list(origin_mut_layer_gen)
                origin_mut_layer_gen = executor.map(gen.mut_45, origin_mut_layer_gen)
                origin_mut_layer_gen = gen.gen_to_list(origin_mut_layer_gen)

            origin_mut_whole = np.unique(origin_mut_whole_gen, axis=0)
            origin_mut_layer = np.unique(origin_mut_layer_gen, axis=0)

            fitness_res_whole_gen = executor.map(gen.Fitness, origin_mut_whole)
            fitness_res_layer_gen = executor.map(gen.Fitness, origin_mut_layer)

            fitness_res_whole, fitness_res_layer = [], []

            for fit_res_layer in fitness_res_layer_gen:
                fitness_res_layer.append(fit_res_layer[:2])

            for fit_res_whole in fitness_res_whole_gen:
                fitness_res_whole.append(fit_res_whole[:2])

            mean_whole = np.mean(fitness_whole)
            mean_345 = np.mean(fitness_345)

            for i in range(len(fitness_res_whole)):
                accept = np.random.rand()
                p_whole = mean_whole/fitness_res_whole[i][0]
                p_layer = mean_345/fitness_res_whole[i][1]
                if (accept < np.min([1, np.exp(p_whole/3)]) #!
                    and accept < np.min([1, np.exp(p_layer/3)])):
                    origin_list = np.insert(np.array(origin_list), len(origin_list), origin_mut_whole[i], axis=0)
                    fitness_whole = np.insert(fitness_whole, len(fitness_whole), fitness_res_whole[i][0])
                    fitness_345 = np.insert(fitness_345, len(fitness_345), fitness_res_whole[i][1])

            for i in range(len(fitness_res_layer)):
                accept = np.random.rand()
                p_whole = mean_whole - fitness_res_layer[i][0] #*Ot - Ot+1
                p_layer = mean_345 - fitness_res_layer[i][1]
                if (accept < np.min([1, np.exp(p_whole/3)]) #!
                    and accept < np.min([1, np.exp(p_layer/3)])):
                    origin_list = np.insert(np.array(origin_list), len(origin_list), origin_mut_layer[i], axis=0)
                    fitness_whole = np.insert(fitness_whole, len(fitness_whole), fitness_res_layer[i][0])
                    fitness_345 = np.insert(fitness_345, len(fitness_345), fitness_res_layer[i][1])

            if kkt % 50 == 0 or kkt == try_step - 1:
                origin_list = np.unique(origin_list, axis=0)
                fitness_whole, fitness_345 = [], []
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_work) as executor:
                    fitness_ratio_gen = executor.map(gen.Fitness, origin_list)
                    
                    for fit_ratio in fitness_ratio_gen:
                        fitness_whole.append(fit_ratio[0])
                        fitness_345.append(fit_ratio[1])

        if len(origin_list) >= 200:
            new_type = []
            # new_type_ind = [i for i in range(len(fitness_whole)) if np.random.rand() < np.min([1, np.exp((np.mean(fitness_whole)-fitness_whole[i])/5)])]
            new_type_ind = np.where(fitness_whole <= np.mean(fitness_whole))[0] #!
            if len(new_type_ind) >= 100:
                for nt in new_type_ind:
                    new_type.append(origin_list[nt].tolist())
                new_type_red = np.unique(np.array(new_type), axis=0)
                if len(new_type_red) >= 60:
                    origin_list = new_type_red.copy()
                    fitness_whole, fitness_345 = [], []
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_work) as executor:
                        fitness_ratio_gen = executor.map(gen.Fitness, origin_list)
                        
                        for fit_ratio in fitness_ratio_gen:
                            fitness_whole.append(fit_ratio[0])
                            fitness_345.append(fit_ratio[1])

        assert len(origin_list) == len(fitness_whole), f'Not same length {len(origin_list)} {len(fitness_whole)}'

        if kkt % 100 == 0:
            print(f'Fitness mean: {np.mean(fitness_whole)} iter {kkt} length of list {len(origin_list)}')
            fitness_mean.append(np.mean(fitness_whole))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        fitness_ratio_gen = executor.map(gen.Fitness, origin_list)

        fitness_ = []
        for fit_ratio in fitness_ratio_gen:
            fitness_.append(fit_ratio[0])

    np.save(f'/media/wz/7AD631A4D6316195/Projects/GSFE/origin_list_qua/origin_list{int(cr_*100)}{int(mn_*100)}{int(co_*100)}.npy', origin_list)
    np.save(f'/media/wz/7AD631A4D6316195/Projects/GSFE/origin_list_qua/fitness_result{int(cr_*100)}{int(mn_*100)}{int(co_*100)}.npy', fitness_)
    np.save(f'/media/wz/7AD631A4D6316195/Projects/GSFE/origin_list_qua/fitness_whole{int(cr_*100)}{int(mn_*100)}{int(co_*100)}.npy', fitness_mean)
    
    second_to_hour(time.time() - start_)
