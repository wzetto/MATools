import math
import json
import numpy as np
from random import randrange
from sub_func_ce import abs_dis, find_overlap
from itertools import combinations, permutations

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
    
class CE:
    def __init__(self, 
        ind_1nn, ind_2nn, ind_3nn, ind_4nn, 
        ind_5nn, ind_6nn, ind_qua1nn, ind_qua1nn2nn,
        ind_qua111122,

        ind_trip111, ind_trip112, ind_trip113, ind_trip114,
        ind_trip123, ind_trip125, ind_trip133, ind_trip134, 
        ind_trip135, ind_trip145, ind_trip155, ind_trip224,
        ind_trip233, ind_trip255, ind_trip334, ind_trip335, 
        ind_trip345, ind_trip444, ind_trip455, 

        ind_1nn_pbc, ind_2nn_pbc, ind_3nn_pbc, ind_4nn_pbc, 
        ind_5nn_pbc, ind_6nn_pbc, ind_qua1nn_pbc, ind_qua1nn2nn_pbc, 
        ind_qua111122_pbc,

        ind_trip111_pbc, ind_trip112_pbc, ind_trip113_pbc, ind_trip114_pbc,
        ind_trip123_pbc, ind_trip125_pbc, ind_trip133_pbc, ind_trip134_pbc, 
        ind_trip135_pbc, ind_trip145_pbc, ind_trip155_pbc, ind_trip224_pbc,
        ind_trip233_pbc, ind_trip255_pbc, ind_trip334_pbc, ind_trip335_pbc, 
        ind_trip345_pbc, ind_trip444_pbc, ind_trip455_pbc, 
        #* 21 quadruplets
        qua_indlist_raw, qua_indlist_pbc, 

        ind_raw,):

        #* Inner of 32-atom config.
        self.ind_1nn = ind_1nn
        self.ind_2nn = ind_2nn
        self.ind_3nn = ind_3nn
        self.ind_4nn = ind_4nn
        self.ind_5nn = ind_5nn
        self.ind_6nn = ind_6nn
        self.ind_qua1nn = ind_qua1nn
        self.ind_qua1nn2nn = ind_qua1nn2nn
        self.ind_qua111122 = ind_qua111122

        #* Cluster on boundary of primary config.
        self.ind_1nn_pbc = ind_1nn_pbc
        self.ind_2nn_pbc = ind_2nn_pbc
        self.ind_3nn_pbc = ind_3nn_pbc
        self.ind_4nn_pbc = ind_4nn_pbc
        self.ind_5nn_pbc = ind_5nn_pbc
        self.ind_6nn_pbc = ind_6nn_pbc
        self.ind_qua1nn_pbc = ind_qua1nn_pbc
        self.ind_qua1nn2nn_pbc = ind_qua1nn2nn_pbc
        self.ind_qua111122_pbc = ind_qua111122_pbc

        #* Triplets.
        self.ind_trip111, self.ind_trip111_pbc = ind_trip111, ind_trip111_pbc 
        self.ind_trip112, self.ind_trip112_pbc = ind_trip112, ind_trip112_pbc 
        self.ind_trip113, self.ind_trip113_pbc = ind_trip113, ind_trip113_pbc 
        self.ind_trip114, self.ind_trip114_pbc = ind_trip114, ind_trip114_pbc 
        self.ind_trip123, self.ind_trip123_pbc = ind_trip123, ind_trip123_pbc 
        self.ind_trip125, self.ind_trip125_pbc = ind_trip125, ind_trip125_pbc 
        self.ind_trip133, self.ind_trip133_pbc = ind_trip133, ind_trip133_pbc 
        self.ind_trip134, self.ind_trip134_pbc = ind_trip134, ind_trip134_pbc 
        self.ind_trip135, self.ind_trip135_pbc = ind_trip135, ind_trip135_pbc 
        self.ind_trip145, self.ind_trip145_pbc = ind_trip145, ind_trip145_pbc 
        self.ind_trip155, self.ind_trip155_pbc = ind_trip155, ind_trip155_pbc 
        self.ind_trip224, self.ind_trip224_pbc = ind_trip224, ind_trip224_pbc 
        self.ind_trip233, self.ind_trip233_pbc = ind_trip233, ind_trip233_pbc 
        self.ind_trip255, self.ind_trip255_pbc = ind_trip255, ind_trip255_pbc 
        self.ind_trip334, self.ind_trip334_pbc = ind_trip334, ind_trip334_pbc 
        self.ind_trip335, self.ind_trip335_pbc = ind_trip335, ind_trip335_pbc 
        self.ind_trip345, self.ind_trip345_pbc = ind_trip345, ind_trip345_pbc 
        self.ind_trip444, self.ind_trip444_pbc = ind_trip444, ind_trip444_pbc
        self.ind_trip455, self.ind_trip455_pbc = ind_trip455, ind_trip455_pbc

        #* Quadruplets
        self.qua_indlist_raw, self.qua_indlist_pbc = qua_indlist_raw, qua_indlist_pbc

        self.ind_raw = ind_raw
        # self.ind_pbc = ind_pbc

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

        elif mode == 'tri1nn':
            mode_cluster = len(cluster)
            uni, count_uni = np.unique(cluster, return_counts=True)
            len_uni = len(uni)
            mode_sym = mode_cluster - len(uni)
            sym_op = self.sym_optri[mode_sym]
        
        elif mode == 'qua1nn': #* 111111
            uni, count_uni = np.unique(cluster, return_counts=True)
            len_uni = len(uni)
            count_uni = np.sort(count_uni)
            len_count = len(count_uni)
            c_ind = len_uni+len_count+count_uni[-1]-count_uni[0]
            sym_op = self.sym_oplist[c_ind]

        elif mode == 'qua1nn2nn': #* 111112
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

            '''
            Square configuration, could be verified by inputing cluster
            [0,1,2,3] and check the number of symmetry operation.
            '''
        elif mode == 'qua111122': 
            sym_mat = np.tile(cluster,(8,1))
            sym_mat[1] = np.concatenate([cluster[1:], cluster[:1]]) #* pi/2
            sym_mat[2] = np.concatenate([cluster[2:], cluster[:2]]) #* pi
            sym_mat[3] = np.concatenate([cluster[3:], cluster[:3]]) #* 3/2*pi
            sym_mat[4][0], sym_mat[4][1], sym_mat[4][2], sym_mat[4][3] = (
                cluster[1], cluster[0], cluster[3], cluster[2]
            ) #* Reflection 1
            sym_mat[5] = np.concatenate([sym_mat[4][2:], sym_mat[4][:2]]) #* Reflection 2
            sym_mat[6] = np.concatenate([sym_mat[5][3:], sym_mat[5][:3]]) #* Centre 1
            sym_mat[7] = np.concatenate([sym_mat[5][1:], sym_mat[5][:1]]) #* Centre 2

            sym_op = len(np.unique(sym_mat, axis=0))

            ''' 
            From now on the coordinates-indices follow a 
            strictly defined ascending order.
            Details can be checked in {qua_ind.py} file.
            '''
        elif (mode == '111133' or mode == '112233'
            or mode == '112334' or mode == '113444'
            or mode == '223334'):
            #* Replace [1][2]
            sym_mat = np.tile(cluster,(2,1))
            sym_mat[1][1], sym_mat[1][2] = sym_mat[0][2], sym_mat[0][1]

            sym_op = len(np.unique(sym_mat, axis=0))

        elif mode == '111224' or mode == '133444':
            #* Replace [2][3]
            sym_mat = np.tile(cluster,(2,1))
            sym_mat[1][2], sym_mat[1][3] = sym_mat[0][3], sym_mat[0][2]

            sym_op = len(np.unique(sym_mat, axis=0))

        elif mode == '111333' or mode == '222444':
            sym_mat = np.tile(cluster,(6,1))
            count = 0
            for i, j, k in permutations(cluster[1:], 3):
                sym_mat[count][1:] = np.array([i, j, k])
                count += 1

            sym_op = len(np.unique(sym_mat, axis=0))

        elif mode == '111334':
            sym_mat = np.tile(cluster,(2,1))
            #* Replace [0][1]
            sym_mat[1][0], sym_mat[1][1] = sym_mat[0][1], sym_mat[0][0]

            sym_op = len(np.unique(sym_mat, axis=0))

        elif mode == '113344':
            sym_mat = np.tile(cluster,(4,1))
            sym_mat[1] = cluster[1], cluster[0], cluster[3], cluster[2]
            sym_mat[2] = cluster[2], cluster[3], cluster[0], cluster[1]
            sym_mat[3] = cluster[3], cluster[2], cluster[1], cluster[0]

            sym_op = len(np.unique(sym_mat, axis=0))

        elif mode == '222244':
            sym_mat = np.tile(cluster,(8,1))
            sym_mat[1] = cluster[0], cluster[2], cluster[1], cluster[3]
            sym_mat[2] = cluster[1], cluster[0], cluster[3], cluster[2]
            sym_mat[3] = cluster[1], cluster[3], cluster[0], cluster[2]
            sym_mat[4] = cluster[2], cluster[0], cluster[3], cluster[1]
            sym_mat[5] = cluster[2], cluster[3], cluster[0], cluster[1]
            sym_mat[6] = cluster[3], cluster[1], cluster[2], cluster[0]
            sym_mat[7] = cluster[3], cluster[2], cluster[1], cluster[0]

            sym_op = len(np.unique(sym_mat, axis=0))

        elif mode == '223333':
            sym_mat = np.tile(cluster,(8,1))
            sym_mat[1] = cluster[0], cluster[3], cluster[2], cluster[1]
            sym_mat[2] = cluster[1], cluster[0], cluster[3], cluster[2]
            sym_mat[3] = cluster[1], cluster[2], cluster[3], cluster[0]
            sym_mat[4] = cluster[2], cluster[1], cluster[0], cluster[3]
            sym_mat[5] = cluster[2], cluster[3], cluster[0], cluster[1]
            sym_mat[6] = cluster[3], cluster[0], cluster[1], cluster[2]
            sym_mat[7] = cluster[3], cluster[2], cluster[1], cluster[0]

            sym_op = len(np.unique(sym_mat, axis=0))

        elif (mode == '111123' or mode == '111134'
            or mode == '111233' or mode == '112234'
            or mode == '112333' or mode == '113334'
            or mode == '122334' or mode == '123333'):

            sym_op = 1

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

    def trip_extract(self, config, ind_trip_raw, ind_trip_pbc, type_trip, cpr):
        #* Merge the pbc and raw indices list.
        ind_trip_all = np.concatenate([ind_trip_raw, ind_trip_pbc], axis=0)

        for i in ind_trip_all:
            a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            cluster = np.array([a1, a2, a3])

            if type_trip == '111':
                cpr += self.cpr(cluster)/self.sym_operator(cluster, mode='tri1nn')

            elif type_trip == '112':
                #* Symmetry operation will be done only in AAB or ABC form
                # print(a1, a2, a3)
                if ((a1 == a2 and a1 != a3) 
                    or (a1 == a3 and a1 != a2) 
                    or (len(np.unique(cluster)) == 3)):
                    cpr += self.cpr(cluster)/2
                else:
                    cpr += self.cpr(cluster)

            elif type_trip == '123':
                cpr += self.cpr(cluster)
        
        cpr = np.array(cpr)/len(ind_trip_all)

        return cpr

    def qua_extract(self, config, ind_qua_raw, ind_qua_pbc, type_qua, cpr=0):
        #* Merge the pbc and raw indices list.
        ind_qua_all = np.concatenate([ind_qua_raw, ind_qua_pbc], axis=0)

        for i in ind_qua_all:
            a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
            cluster = np.array([a1, a2, a3, a4])

            cpr += self.cpr(cluster)/self.sym_operator(cluster, mode=type_qua)

        cpr = np.array(cpr)/len(ind_qua_all)

        return cpr

    def cluster_extra(self, config, embed):
        '''
        Config in PBC must be in R = (N*27)x3 form.
        '''
        cpr_1nn = 0
        if embed['pair1']:
            for i in self.ind_1nn:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_1nn += (self.cpr(cluster)/self.sym_operator(cluster))

            for i in self.ind_1nn_pbc:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_1nn += (self.cpr(cluster)/self.sym_operator(cluster))  
            
            cpr_1nn = np.array(cpr_1nn)/(len(self.ind_1nn)+len(self.ind_1nn_pbc))
        else:
            cpr_1nn = []

        cpr_2nn = 0
        if embed['pair2']:
            for i in self.ind_2nn:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_2nn += (self.cpr(cluster)/self.sym_operator(cluster))

            for i in self.ind_2nn_pbc:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_2nn += (self.cpr(cluster)/self.sym_operator(cluster))

            cpr_2nn = np.array(cpr_2nn)/(len(self.ind_2nn)+len(self.ind_2nn_pbc))
        else:
            cpr_2nn = []
        
        cpr_3nn = 0
        if embed['pair3']:
            for i in self.ind_3nn:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_3nn += (self.cpr(cluster)/self.sym_operator(cluster))

            for i in self.ind_3nn_pbc:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_3nn += (self.cpr(cluster)/self.sym_operator(cluster))

            cpr_3nn = np.array(cpr_3nn)/(len(self.ind_3nn)+len(self.ind_3nn_pbc))
        else:
            cpr_3nn = []

        cpr_4nn = 0
        if embed['pair4']:
            for i in self.ind_4nn:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_4nn += (self.cpr(cluster)/self.sym_operator(cluster))

            for i in self.ind_4nn_pbc:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_4nn += (self.cpr(cluster)/self.sym_operator(cluster))

            cpr_4nn = np.array(cpr_4nn)/(len(self.ind_4nn)+len(self.ind_4nn_pbc))
        else:
            cpr_4nn = []

        cpr_5nn = 0
        if embed['pair5']:
            for i in self.ind_5nn:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_5nn += (self.cpr(cluster)/self.sym_operator(cluster))

            for i in self.ind_5nn_pbc:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_5nn += (self.cpr(cluster)/self.sym_operator(cluster))

            cpr_5nn = np.array(cpr_5nn)/(len(self.ind_5nn)+len(self.ind_5nn_pbc))
        else:
            cpr_5nn = []

        cpr_6nn = 0
        if embed['pair6']:
            for i in self.ind_6nn:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_6nn += (self.cpr(cluster)/self.sym_operator(cluster))

            for i in self.ind_6nn_pbc:
                a1, a2 = config[i[0]], config[i[1]]
                cluster = np.array([a1, a2])
                cpr_6nn += (self.cpr(cluster)/self.sym_operator(cluster))

            cpr_6nn = np.array(cpr_6nn)/(len(self.ind_6nn)+len(self.ind_6nn_pbc))
        else:
            cpr_6nn = []
            
        cpr_tri111 = 0
        if embed['tri111']:
            cpr_tri111 = self.trip_extract(config, self.ind_trip111,
                self.ind_trip111_pbc, '111', cpr_tri111)
        else:
            cpr_tri111 = []

        cpr_tri444 = 0
        if embed['tri444']:
            cpr_tri444 = self.trip_extract(config, self.ind_trip444,
                self.ind_trip444_pbc, '111', cpr_tri444)
        else:
            cpr_tri444 = []
            # for i in self.ind_trip1nn:
            #     a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            #     cluster = np.array([a1, a2, a3])
            #     cpr_tri1nn += (self.cpr(cluster)/self.sym_operator(cluster, mode='tri1nn')).tolist()

            # for i in self.ind_trip1nn_pbc:
            #     a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            #     cluster = np.array([a1, a2, a3])
            #     cpr_tri1nn += (self.cpr(cluster)/self.sym_operator(cluster, mode='tri1nn')).tolist()

            # cpr_tri1nn = np.array(cpr_tri1nn)/(len(self.ind_trip1nn)+len(self.ind_trip1nn_pbc))

        cpr_tri112 = 0
        if embed['tri112']:
            cpr_tri112 = self.trip_extract(config, self.ind_trip112,
                self.ind_trip112_pbc, '112', cpr_tri112)
        else:
            cpr_tri112 = []
        
        cpr_tri113 = 0
        if embed['tri113']:
            cpr_tri113 = self.trip_extract(config, self.ind_trip113,
                self.ind_trip113_pbc, '112', cpr_tri113)
        else:
            cpr_tri113 = []

        cpr_tri114 = 0
        if embed['tri114']:
            cpr_tri114 = self.trip_extract(config, self.ind_trip114,
                self.ind_trip114_pbc, '112', cpr_tri114)
        else:
            cpr_tri114 = []
        
        cpr_tri133 = 0
        if embed['tri133']:
            cpr_tri133 = self.trip_extract(config, self.ind_trip133,
                self.ind_trip133_pbc, '112', cpr_tri133)
        else:
            cpr_tri133 = []

        cpr_tri155 = 0
        if embed['tri155']:
            cpr_tri155 = self.trip_extract(config, self.ind_trip155,
                self.ind_trip155_pbc, '112', cpr_tri155)
        else:
            cpr_tri155 = []

        cpr_tri224 = 0
        if embed['tri224']:
            cpr_tri224 = self.trip_extract(config, self.ind_trip224,
                self.ind_trip224_pbc, '112', cpr_tri224)
        else:
            cpr_tri224 = []
        
        cpr_tri233 = 0
        if embed['tri233']:
            cpr_tri233 = self.trip_extract(config, self.ind_trip233,
                self.ind_trip233_pbc, '112', cpr_tri233)
        else:
            cpr_tri233 = []

        cpr_tri255 = 0
        if embed['tri255']:
            cpr_tri255 = self.trip_extract(config, self.ind_trip255,
                self.ind_trip255_pbc, '112', cpr_tri255)
        else:
            cpr_tri255 = []

        cpr_tri334 = 0
        if embed['tri334']:
            cpr_tri334 = self.trip_extract(config, self.ind_trip334,
                self.ind_trip334_pbc, '112', cpr_tri334)
        else:
            cpr_tri334 = []

        cpr_tri335 = 0
        if embed['tri335']:
            cpr_tri335 = self.trip_extract(config, self.ind_trip335,
                self.ind_trip335_pbc, '112', cpr_tri335)
        else:
            cpr_tri335 = []

        cpr_tri455 = 0
        if embed['tri455']:
            cpr_tri455 = self.trip_extract(config, self.ind_trip455,
                self.ind_trip455_pbc, '112', cpr_tri455)
        else:
            cpr_tri455 = []
            # for i in self.ind_trip1nn2nn_pbc:
            #     a1, a2, a3 = config[i[0]], config[i[1]], config[i[2]]
            #     cluster = np.array([a1, a2, a3])
            #     #* Symmetry operation will be doen only in AAB or ABC form
            #     if ((a1 == a2 and a1 != a3) 
            #         or (a1 == a3 and a1 != a2) 
            #         or (np.unique(cluster) == 3)):
            #         cpr_tri1nn2nn += (self.cpr(cluster)/2).tolist()
            #     else:
            #         cpr_tri1nn2nn += self.cpr(cluster).tolist()

            # cpr_tri1nn2nn = np.array(cpr_tri1nn2nn)/(len(self.ind_trip1nn2nn)+len(self.ind_trip1nn2nn_pbc))

        cpr_tri123 = 0
        if embed['tri123']:
            cpr_tri123 = self.trip_extract(config, self.ind_trip123,
                self.ind_trip123_pbc, '123', cpr_tri123)
        else:
            cpr_tri123 = []

        cpr_tri125 = 0
        if embed['tri125']:
            cpr_tri125 = self.trip_extract(config, self.ind_trip125,
                self.ind_trip125_pbc, '123', cpr_tri125)
        else:
            cpr_tri125 = []

        cpr_tri134 = 0
        if embed['tri134']:
            cpr_tri134 = self.trip_extract(config, self.ind_trip134,
                self.ind_trip134_pbc, '123', cpr_tri134)
        else:
            cpr_tri134 = []

        cpr_tri135 = 0
        if embed['tri135']:
            cpr_tri135 = self.trip_extract(config, self.ind_trip135,
                self.ind_trip135_pbc, '123', cpr_tri135)
        else:
            cpr_tri135 = []

        cpr_tri145 = 0
        if embed['tri145']:
            cpr_tri145 = self.trip_extract(config, self.ind_trip145,
                self.ind_trip145_pbc, '123', cpr_tri145)
        else:
            cpr_tri145 = []

        cpr_tri345 = 0
        if embed['tri345']:
            cpr_tri345 = self.trip_extract(config, self.ind_trip345,
                self.ind_trip345_pbc, '123', cpr_tri345)
        else:
            cpr_tri345 = []

        cpr_qua1nn = 0
        if embed['qua111111']:
            for i in self.ind_qua1nn:
                a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
                cluster = np.array([a1, a2, a3, a4])
                cpr_qua1nn += (self.cpr(cluster)/self.sym_operator(cluster, mode='qua1nn'))

            for i in self.ind_qua1nn_pbc:
                a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
                cluster = np.array([a1, a2, a3, a4])
                cpr_qua1nn += (self.cpr(cluster)/self.sym_operator(cluster, mode='qua1nn'))

            cpr_qua1nn = np.array(cpr_qua1nn)/(len(self.ind_qua1nn) + len(self.ind_qua1nn_pbc))
        else:
            cpr_qua1nn = []

        cpr_qua1nn2nn = 0
        if embed['qua111112']:
            for i in self.ind_qua1nn2nn:
                a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
                cluster = np.array([a1, a2, a3, a4])
                cpr_qua1nn2nn += (self.cpr(cluster)/self.sym_operator(cluster, mode='qua1nn2nn'))

            for i in self.ind_qua1nn2nn_pbc:
                a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
                cluster = np.array([a1, a2, a3, a4])
                cpr_qua1nn2nn += (self.cpr(cluster)/self.sym_operator(cluster, mode='qua1nn2nn'))

            cpr_qua1nn2nn = np.array(cpr_qua1nn2nn)/(len(self.ind_qua1nn2nn)+len(self.ind_qua1nn2nn_pbc))
        else:
            cpr_qua1nn2nn = []

        cpr_qua111122 = 0
        if embed['qua111122']:
            for i in self.ind_qua111122:
                a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
                cluster = np.array([a1, a2, a3, a4])
                cpr_qua111122 += (self.cpr(cluster)/self.sym_operator(cluster, mode='qua111122'))

            for i in self.ind_qua111122_pbc:
                a1, a2, a3, a4 = config[i[0]], config[i[1]], config[i[2]], config[i[3]]
                cluster = np.array([a1, a2, a3, a4])
                cpr_qua111122 += (self.cpr(cluster)/self.sym_operator(cluster, mode='qua111122'))

            cpr_qua111122 = np.array(cpr_qua111122)/(len(self.ind_qua111122)+len(self.ind_qua111122_pbc))
        else:
            cpr_qua111122 = []

        '''
        The cpr matrix for 21 quadruplet clusters
        '''
        cpr_quaremain = np.zeros((21, 81))
        qua_remain_embed = [
            '111123', '111133', '111134', '111224',
            '111233', '111333', '111334', '112233',
            '112234', '112333', '112334', '113334',
            '113344', '113444', '122334', '123333',
            '133444', '222244', '222444', '223333', '223334',
        ]
        effect_qua_ind = []
        for i_ in range(len(qua_remain_embed)):
            qua_type = qua_remain_embed[i_]
            if embed[qua_type]:

                q1,q2,q3,q4,q5,q6 = qua_type
                cluster_type_raw = f'{q1}nn{q2}nn{q3}nn{q4}nn{q5}nn{q6}nn_raw'
                cluster_type_pbc = f'{q1}nn{q2}nn{q3}nn{q4}nn{q5}nn{q6}nn_pbc'

                cpr_quaremain[i_] = self.qua_extract(config, 
                    #* Dicts of indices for qua. clusters.
                    self.qua_indlist_raw[cluster_type_raw], 
                    self.qua_indlist_pbc[cluster_type_pbc], 
                    qua_type)

                effect_qua_ind.append(i_)

        #* R = 1 x (N x 81)
        cpr_qua_remain = cpr_quaremain[effect_qua_ind].flatten()

        #* Return the concatenate array (R = 1 x sum(M))
        return np.concatenate([
            cpr_1nn, cpr_2nn, cpr_3nn, cpr_4nn,
            cpr_tri111, cpr_tri112, cpr_tri113,
            cpr_tri114, cpr_tri123, cpr_tri125,
            cpr_tri133, cpr_tri134, cpr_tri135,
            cpr_tri145, cpr_tri155, cpr_tri224, 
            cpr_tri233, cpr_tri255, cpr_tri334,
            cpr_tri335, cpr_tri345, cpr_tri444, cpr_tri455, 
            cpr_qua1nn, cpr_qua1nn2nn, cpr_qua111122,
            cpr_5nn, cpr_6nn, cpr_qua_remain
        ], 0)