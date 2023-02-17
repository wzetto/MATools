from ce_main import ele_list_gen, CE
import multiprocessing as mp
import argparse
import numpy as np
import json
import time

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

def main(i):
    ele_list = ele_list_gen(a1, a2, a3, a4, 32)
    weight_list_predict = (ce_.cluster_extra(
        np.tile(ele_list, 27), embed_list))

    if i % 100 == 0:
        print(f'iter{i} finished.')

    return weight_list_predict.flatten()

if __name__ == '__main__':
    #* Deducing step
    ind_1nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_1nn.npy')
    ind_2nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_2nn.npy')
    ind_3nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_3nn.npy')
    ind_4nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_4nn.npy')
    ind_5nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_5nn.npy')
    ind_6nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_6nn.npy')
    ind_qua1nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_qua1nn.npy')
    ind_qua1nn2nn = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_qua1nn2nn.npy')
    ind_qua111122 = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_qua111122_raw.npy')

    ind_1nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_1nn_pbc.npy')
    ind_2nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_2nn_pbc.npy')
    ind_3nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_3nn_pbc.npy')
    ind_4nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_4nn_pbc.npy')
    ind_5nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_5nn_pbc.npy')
    ind_6nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_6nn_pbc.npy')
    ind_qua1nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_qua1nn_pbc.npy')
    ind_qua1nn2nn_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_qua1nn2nn_pbc.npy')
    ind_qua111122_pbc = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_qua111122_pbc.npy')

    ''' 
    Thanks chatGPT for doing such *** things.
    '''
    pth_trip = '/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/triplet_ind/'

    ind_trip111 = np.load(f'{pth_trip}ind_trip111_raw.npy')
    ind_trip111_pbc = np.load(f'{pth_trip}ind_trip111_pbc.npy')
    ind_trip112 = np.load(f'{pth_trip}ind_trip112_raw.npy')
    ind_trip112_pbc = np.load(f'{pth_trip}ind_trip112_pbc.npy')
    ind_trip113 = np.load(f'{pth_trip}ind_trip113_raw.npy')
    ind_trip113_pbc = np.load(f'{pth_trip}ind_trip113_pbc.npy')
    ind_trip114 = np.load(f'{pth_trip}ind_trip114_raw.npy')
    ind_trip114_pbc = np.load(f'{pth_trip}ind_trip114_pbc.npy')
    ind_trip123 = np.load(f'{pth_trip}ind_trip123_raw.npy')
    ind_trip123_pbc = np.load(f'{pth_trip}ind_trip123_pbc.npy')
    ind_trip125 = np.load(f'{pth_trip}ind_trip125_raw.npy')
    ind_trip133 = np.load(f'{pth_trip}ind_trip133_raw.npy')
    ind_trip134 = np.load(f'{pth_trip}ind_trip134_raw.npy')
    ind_trip125_pbc = np.load(f'{pth_trip}ind_trip125_pbc.npy')
    ind_trip133_pbc = np.load(f'{pth_trip}ind_trip133_pbc.npy')
    ind_trip134_pbc = np.load(f'{pth_trip}ind_trip134_pbc.npy')
    ind_trip135 = np.load(f'{pth_trip}ind_trip135_raw.npy')
    ind_trip145 = np.load(f'{pth_trip}ind_trip145_raw.npy')
    ind_trip155 = np.load(f'{pth_trip}ind_trip155_raw.npy')
    ind_trip135_pbc = np.load(f'{pth_trip}ind_trip135_pbc.npy')
    ind_trip145_pbc = np.load(f'{pth_trip}ind_trip145_pbc.npy')
    ind_trip155_pbc = np.load(f'{pth_trip}ind_trip155_pbc.npy')
    ind_trip224 = np.load(f'{pth_trip}ind_trip224_raw.npy')
    ind_trip233 = np.load(f'{pth_trip}ind_trip233_raw.npy')
    ind_trip255 = np.load(f'{pth_trip}ind_trip255_raw.npy')
    ind_trip224_pbc = np.load(f'{pth_trip}ind_trip224_pbc.npy')
    ind_trip233_pbc = np.load(f'{pth_trip}ind_trip233_pbc.npy')
    ind_trip255_pbc = np.load(f'{pth_trip}ind_trip255_pbc.npy')
    ind_trip334 = np.load(f'{pth_trip}ind_trip334_raw.npy')
    ind_trip335 = np.load(f'{pth_trip}ind_trip335_raw.npy')
    ind_trip345 = np.load(f'{pth_trip}ind_trip345_raw.npy')
    ind_trip334_pbc = np.load(f'{pth_trip}ind_trip334_pbc.npy')
    ind_trip335_pbc = np.load(f'{pth_trip}ind_trip335_pbc.npy')
    ind_trip345_pbc = np.load(f'{pth_trip}ind_trip345_pbc.npy')
    ind_trip444 = np.load(f'{pth_trip}ind_trip444_raw.npy')
    ind_trip455 = np.load(f'{pth_trip}ind_trip455_raw.npy')
    ind_trip444_pbc = np.load(f'{pth_trip}ind_trip444_pbc.npy')
    ind_trip455_pbc = np.load(f'{pth_trip}ind_trip455_pbc.npy')

    #* 21 x quadruplets
    pth_sav = '/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/MATools/CE_MC/runs/demo/20221216_msadGA/'

    with open(f'{pth_sav}ind_quapbc.json', 'r') as f:
        ind_quapbc = json.load(f)

    with open(f'{pth_sav}ind_quaraw.json', 'r') as f:
        ind_quaraw = json.load(f)

    ind_raw = np.load('/media/wz/7AD631A4D6316195/Projects/mc_pure_qua/fcc_32/ind_raw32.npy')

    ce_ = CE(ind_1nn, ind_2nn, ind_3nn, ind_4nn, ind_5nn, ind_6nn,
            ind_qua1nn, ind_qua1nn2nn, ind_qua111122,

            ind_trip111, ind_trip112, ind_trip113, ind_trip114,
            ind_trip123, ind_trip125, ind_trip133, ind_trip134, 
            ind_trip135, ind_trip145, ind_trip155, ind_trip224,
            ind_trip233, ind_trip255, ind_trip334, ind_trip335, 
            ind_trip345, ind_trip444, ind_trip455, 

            ind_1nn_pbc, ind_2nn_pbc, ind_3nn_pbc, ind_4nn_pbc, 
            ind_5nn_pbc, ind_6nn_pbc,
            ind_qua1nn_pbc, ind_qua1nn2nn_pbc, ind_qua111122_pbc,

            ind_trip111_pbc, ind_trip112_pbc, ind_trip113_pbc, ind_trip114_pbc,
            ind_trip123_pbc, ind_trip125_pbc, ind_trip133_pbc, ind_trip134_pbc, 
            ind_trip135_pbc, ind_trip145_pbc, ind_trip155_pbc, ind_trip224_pbc,
            ind_trip233_pbc, ind_trip255_pbc, ind_trip334_pbc, ind_trip335_pbc, 
            ind_trip345_pbc, ind_trip444_pbc, ind_trip455_pbc, 

            ind_quaraw, ind_quapbc, ind_raw,)

    embed_val = np.ones(50).astype(bool)

    embed_type = [
        'pair1', 'pair2', 'pair3', 'pair4', 
        'tri111', 'tri112', 'tri113', 'tri114',
        'tri123', 'tri125', 'tri133', 'tri134',
        'tri135', 'tri145', 'tri155', 'tri224',
        'tri233', 'tri255', 'tri334', 'tri335',
        'tri345', 'tri444', 'tri455',
        'qua111111', 'qua111112', 'qua111122',
        'pair5', 'pair6',
        #? Quadruplets x 21
        '111123', '111133', '111134', '111224',
        '111233', '111333', '111334', '112233',
        '112234', '112333', '112334', '113334',
        '113344', '113444', '122334', '123333',
        '133444', '222244', '222444', '223333', '223334','point']

    #* Lookup table of the indices in weight for each cluster.
    # embed_book = [
    #     [i for i in range(0,9)],
    #     [i for i in range(9,18)],
    #     [i for i in range(18,27)],
    #     [i for i in range(27,36)],
    #     [i for i in range(36,63)],
    #     [i for i in range(63,90)],
    #     [i for i in range(90,117)],
    #     [i for i in range(117,144)],
    #     [i for i in range(144,171)],
    #     [i for i in range(171,198)],
    #     [i for i in range(198,225)],
    #     [i for i in range(225,252)],
    #     [i for i in range(252,279)],
    #     [i for i in range(279,306)],
    #     [i for i in range(306,333)],
    #     [i for i in range(333,360)],
    #     [i for i in range(360,387)],
    #     [i for i in range(387,414)],
    #     [i for i in range(414,441)],
    #     [i for i in range(441,468)],
    #     [i for i in range(468,495)],
    #     [i for i in range(495,522)],
    #     [i for i in range(522,549)],
    #     [i for i in range(549,630)],
    #     [i for i in range(630,711)],
    #     [i for i in range(711, 792)],
    #     [i for i in range(792, 801)],
    #     [i for i in range(801, 810)],
    # ]

    # qua_embed = np.linspace(810, 810+81*21-1, 81*21).reshape(21, 81).astype(int).tolist()
    # embed_book += qua_embed
#* Considering the cluster function's symmetry
    embed_book = np.load(
        '/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/MATools/CE_MC/runs/demo/20230117_basis_cluster/embed_book_new_230212.npy',
        allow_pickle = True)
    embed_list = dict(zip(embed_type, embed_val))

    ce_ = CE(ind_1nn, ind_2nn, ind_3nn, ind_4nn, ind_5nn, ind_6nn,
        ind_qua1nn, ind_qua1nn2nn, ind_qua111122,

        ind_trip111, ind_trip112, ind_trip113, ind_trip114,
        ind_trip123, ind_trip125, ind_trip133, ind_trip134, 
        ind_trip135, ind_trip145, ind_trip155, ind_trip224,
        ind_trip233, ind_trip255, ind_trip334, ind_trip335, 
        ind_trip345, ind_trip444, ind_trip455, 

        ind_1nn_pbc, ind_2nn_pbc, ind_3nn_pbc, ind_4nn_pbc, 
        ind_5nn_pbc, ind_6nn_pbc,
        ind_qua1nn_pbc, ind_qua1nn2nn_pbc, ind_qua111122_pbc,

        ind_trip111_pbc, ind_trip112_pbc, ind_trip113_pbc, ind_trip114_pbc,
        ind_trip123_pbc, ind_trip125_pbc, ind_trip133_pbc, ind_trip134_pbc, 
        ind_trip135_pbc, ind_trip145_pbc, ind_trip155_pbc, ind_trip224_pbc,
        ind_trip233_pbc, ind_trip255_pbc, ind_trip334_pbc, ind_trip335_pbc, 
        ind_trip345_pbc, ind_trip444_pbc, ind_trip455_pbc, 

        ind_quaraw, ind_quapbc, ind_raw,)

    #* Input the atomic content
    start_ = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--atomic_ratios', type=float, nargs=4, 
    required=True)
    parser.add_argument('--max_processor', type=int, default=20)
    parser.add_argument('--num_config', type=int, default=1000)
    args = parser.parse_args()

    #* Multi-processing part
    a1, a2, a3, a4 = args.atomic_ratios
    pool = mp.Pool(processes = args.max_processor)
    outputs = pool.map(main, range(args.num_config))
    pool.close()
    pool.join()

    #* Save results
    config_list = []

    at_ratio = f'{int(a1*100)}_{int(a2*100)}_{int(a3*100)}_mergebasis_50cluster'
    pth = '/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/MATools_buffer/msadGA/202212/configs/'

    for output in outputs:
        config_list.append(output)
    
    second_to_hour(time.time()-start_)

    np.save(f'{pth}{at_ratio}', config_list)