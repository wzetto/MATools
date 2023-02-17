import numpy as np
import re

split_a = -1
split_b = -4

def msad_ana_qua(cont_dir, pos_dir, lattice_param, cell_num):
    param = {}
    lattice_param = lattice_param * cell_num

    li_sum_cont, strip_cont, cont_coor = get_coor(cont_dir)
    li_sum_pos, strip_pos, pos_coor = get_coor(pos_dir)
    # print(f'Num of Cr, Mn, Co, Ni: {strip_pos}')
    param['Total_num'] = li_sum_pos
    param['CrMnCoNi'] = strip_pos

    cont_coor = cont_coor * lattice_param #Lattice parameter * cell units
    pos_coor = pos_coor * lattice_param

    param['Residual'] = cont_coor - pos_coor
    #* 10000 means convert angstorm into picometer
    #* li_sum_pos means total number of atoms
    divide = 10000 / li_sum_pos
    cr_num, mn_num, co_num, ni_num = int(strip_pos[0]), int(strip_pos[1]), int(strip_pos[2]), int(strip_pos[3])
    divide_cr, divide_mn, divide_co, divide_ni = 10000 / cr_num, 10000 / mn_num, 10000 / co_num, 10000 / ni_num, 
#     print(cont_coor)
    msad = np.sum(np.power((cont_coor - pos_coor), 2)) * divide
    msad_cr = np.sum(np.power(cont_coor[:cr_num] - pos_coor[:cr_num], 2)) * divide_cr
    msad_mn = np.sum(np.power(cont_coor[cr_num:cr_num+mn_num] - pos_coor[cr_num:cr_num+mn_num], 2)) *divide_mn
    msad_co = np.sum(np.power(cont_coor[cr_num+mn_num:cr_num+mn_num+co_num] 
                              - pos_coor[cr_num+mn_num:cr_num+mn_num+co_num], 2)) * divide_co
    msad_ni = np.sum(np.power(cont_coor[cr_num+mn_num+co_num:] - pos_coor[cr_num+mn_num+co_num:], 2)) * divide_ni

    #* Extract coordinates for elements
    param['Cr_coord'] = pos_coor[:cr_num]
    param['Co_coord'] = pos_coor[cr_num:cr_num+mn_num]
    param['Mn_coord'] = pos_coor[cr_num+mn_num:cr_num+mn_num+co_num]
    param['Ni_coord'] = pos_coor[cr_num+mn_num+co_num:]

    param['MSAD'] = round(msad,4)
    param['cr_msad'] = round(msad_cr, 4)
    param['mn_msad'] = round(msad_mn, 4)
    param['co_msad'] = round(msad_co, 4)
    param['ni_msad'] = round(msad_ni, 4)
    
    return param

def read(incar_dir):
    with open(incar_dir) as f:
        input_strip = [s.strip() for s in f.readlines()]
    return input_strip

def get_coor(input_dir, tline = 8):
    with open(input_dir) as f:
        input_strip = [s.strip() for s in f.readlines()]
    
    def trans2list(i):
        s = ''.join(input_strip[i])
        strip = s.split()
        strip = [float(x) for x in strip[:3]]
        return strip

    s = ''.join(input_strip[6])
    strip_6 = s.split()
    strip_6 = [float(x) for x in strip_6]
    li_sum = int(sum(strip_6))
    split_pattern = '[/\\\]'
    type_dir = re.split(split_pattern, input_dir)[-1]
    # print(f'Num of atoms in {type_dir}: {li_sum}')

    input_coor = []

    for i in range(tline, tline + li_sum):
        strip = trans2list(i)
        input_coor += strip

    input_coor = np.array(input_coor).reshape(li_sum, 3)

    assert len(input_coor) == li_sum

    return li_sum, strip_6, input_coor