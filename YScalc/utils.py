import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from scipy.interpolate import interp1d
from collections import defaultdict
from scipy.interpolate import make_interp_spline
from glob import glob
from os.path import join,relpath
import re
from natsort import natsorted
from scipy.signal import savgol_filter
import time
import math
import matplotlib.ticker

def remove_nan(mylist):
    newlist = [x for x in mylist if math.isnan(x) == False]
    return newlist

def mpdata_duplicates(seq):
    resultstraintally = defaultdict(list)
    for i,item in enumerate(seq):
        resultstraintally[item].append(i)
    return ((locs) for key,locs in resultstraintally.items() 
                            if len(locs)>1)

def read_csv_eng(file, Thickness5, Width5, l):
    s5 = Thickness5*Width5
    mp5 = pd.read_csv(file, header = 0, names = ["Stress", "Strain"], skiprows = [1,2,3], encoding= 'unicode_escape')
    mp5['StrainTrue'] = mp5['Strain']/l*100
    mp5['StressTrue'] = mp5['Stress']/s5*1000
    strain = mp5['StrainTrue']
    stress = mp5['StressTrue']
    mp5_strain = strain.values.tolist()
    resultstrain = []
    mp5_stress = stress.values.tolist()
    resultstress = []
    for s_strain in mp5_strain:
        resultstrain.append(s_strain)
    for s_stress in mp5_stress:
        resultstress.append(s_stress)
    print(f'UTS: {np.max(resultstress)}MPa')
    print(f'Elongation: {np.max(resultstrain)}%')
    
    return resultstrain, resultstress

def merge_list(list1):
    return sum(list1,[])

def np_move_avg(a, n, mode = 'valid'):
    return(np.convolve(a, np.ones((n,))/n, mode = mode))

def merge_file(pathoa, extension):
    files = [relpath(x,pathoa) for x in sorted(glob(join(pathoa,'*.{}'.format(extension))))]
    csvfiles = []
    i = 0
    while i < len(files):
        csvfiles.append(join(pathoa, files[i]))
        i += 1
    csvfiles = natsorted(csvfiles)
    return csvfiles

def resultdata_gen(csvfiles, Thickness5, Width5, l):
    resultstrain, resultstress = [], []

    for file in csvfiles:
        resultstrain.append(read_csv_eng(file, Thickness5, Width5, l)[0])
        resultstress.append(read_csv_eng(file, Thickness5, Width5, l)[1])

    resultstrain = merge_list(resultstrain)
    resultstress = merge_list(resultstress)
    return resultstrain, resultstress

    print(f'UTS: {np.max(resultstress)}MPa')
    print(f'Elongation: {np.max(resultstrain)}%')

def convol_smooth(resultstraink, resultstressk, cut_pt, convol_N):
    maxindex = resultstressk.index(np.max(resultstressk))

    resultstrain1 = resultstraink
    resultstress1 = resultstressk
    resultstrainlast = resultstraink[maxindex+1:]
    resultstresslast = resultstressk[maxindex+1:]
    
    resultstress_smooth = resultstressk[:cut_pt]
    resultstress_smooth1 = np_move_avg(resultstressk[cut_pt:], convol_N, mode = 'valid')
    resultstress_smooth1 = resultstress_smooth1.tolist()

    resultstrain_smooth = resultstraink[:cut_pt]
    resultstrain_smooth1 = np.linspace(resultstraink[cut_pt], resultstraink[-1], len(resultstress_smooth1))
    resultstrain_smooth1 = resultstrain_smooth1.tolist()

    resultstrain_smooth.extend(resultstrain_smooth1)
    resultstress_smooth.extend(resultstress_smooth1)
    
    return resultstrain1, resultstress1, resultstrain_smooth, resultstress_smooth

def interp1d_smooth(resultstrain, resultstress, len_multi):
    duplist = [i for i in mpdata_duplicates(resultstrain)]
    icut = 0
    duplistcut = []
    while icut <= len(duplist) - 1:
        duplistcut.append(duplist[icut][1:])
        icut += 1
    dupmerge = merge_list(duplistcut)
    i = 0
    while i <= len(dupmerge) - 1:
        del resultstrain[dupmerge[i]]
        resultstrain.insert(dupmerge[i], 8888)
        del resultstress[dupmerge[i]]
        resultstress.insert(dupmerge[i], 8888)
        i += 1
    #del resultstrain[dupmerge[i]

    resultstrain = list(filter(lambda a: a != 0, resultstrain))
    resultstress = list(filter(lambda a: a != 0, resultstress))
    #remove 888
    while 8888 in resultstrain:
        resultstrain.remove(8888)
    while 8888 in resultstress:
        resultstress.remove(8888)

    resultstrain.sort()
    resultstress.sort()
    if len(resultstrain) < len(resultstress):
        resultstrain.insert(0,0)

    strainsmooth = np.array(resultstrain)
    stresssmooth = np.array(resultstress)
    cubic_interploation_model=interp1d(strainsmooth,stresssmooth,kind="cubic")
    lensmooth = len(resultstrain)*len_multi
    resultstrain1 = np.linspace(resultstrain[0],resultstrain[-1],lensmooth)
    resultstress1 = make_interp_spline(strainsmooth, stresssmooth)(resultstrain1)#平滑设置
    
    return resultstrain1, resultstress1

def elastic_main_plot(sp, fit_len, step, cutfr, range_op, range_ed, resultstrain1, resultstress1, dis_x, dis_y, sample_name,
                     resultstrain_plot, resultstress_plot):
    start_ = time.time()

    n = []
    #sp = 10000 #start point
    #fit_len = 20000 #fit length
    #step = 25 #scan step
    #cutfr = 15 #length percentage of plastic deformation stage
    i = sp
    k1 = fit_len + 1
    len2 = int(len(resultstrain1)/14)
    len1 = len2 - k1
    while i <= len1:
        i2 = i + fit_len
        xi = resultstrain1[i:i2]
        yi = resultstress1[i:i2]
        xi = np.array(xi)
        yi = np.array(yi)
        x_bar = np.mean(xi)
        y_bar = np.mean(yi)
        Sxx = np.sum((xi - x_bar)**2)
        Sxy = np.sum((xi - x_bar)*(yi - y_bar))
        b1_hat = Sxy/Sxx
        b0_hat = y_bar - b1_hat*x_bar
        R_num = np.sum((b0_hat + b1_hat*xi - y_bar)**2)
        R_den = np.sum((yi - y_bar)**2)
        R2_hat = R_num/R_den
        n.append(R2_hat)
        i += step
    index = n.index(max(n))
    imax = sp + index*step
    imax2 = imax + fit_len
    lenfr = int(len2/cutfr)
    print(f'R2_hat: {R2_hat}')

    #Draw the stage 1 curve
    len21 = int(len2)
    xi = resultstrain1[imax:imax2]
    yi = resultstress1[imax:imax2]
    xi = np.array(xi)
    yi = np.array(yi)
    x_bar = np.mean(xi)
    y_bar = np.mean(yi)
    Sxx = np.sum((xi - x_bar)**2)
    Sxy = np.sum((xi - x_bar)*(yi - y_bar))
    b1_hat = Sxy/Sxx
    b0_hat = y_bar - b1_hat*x_bar
    R_num = np.sum((b0_hat + b1_hat*xi - y_bar)**2)
    R_den = np.sum((yi - y_bar)**2)
    R2_hat = R_num/R_den
    xdat = np.linspace(range_op, range_ed, len21)
    y_hat = b0_hat + b1_hat*xdat
    xdat2 = xdat + 0.2
    #plt.plot(xdat, y_hat)

    #Draw the stage 2 curve
    lenf = len2
    #resultf = int((resultstrain[-1] - resultstrain[0]) * 3 / 4)
    xif = resultstrain1[-lenfr:]
    yif = resultstress1[-lenfr:]
    xif = np.array(xif)
    yif = np.array(yif)
    x_barf = np.mean(xif)
    y_barf = np.mean(yif)
    Sxxf = np.sum((xif - x_barf)**2)
    Sxyf = np.sum((xif - x_barf)*(yif - y_barf))
    b1_hatf = Sxyf/Sxxf
    b0_hatf = y_barf - b1_hatf*x_barf
    R_numf = np.sum((b0_hatf + b1_hatf*xif - y_barf)**2)
    R_denf = np.sum((yif - y_barf)**2)
    R2_hatf = R_numf/R_denf
    xdatf = np.linspace(range_op, range_ed, lenf)
    y_hatf = b0_hatf + b1_hatf*xdatf

    #Deduce the cross point for the 0.2% proof stress
    resultstrainmax = resultstrain1[-1] - 0.2
    y_hatmax = b0_hat + b1_hat*resultstrainmax
    resultstrainmove = np.array(resultstrain1)
    y_hatmove = b0_hat + b1_hat*resultstrainmove - b1_hat/5
    y = y_hatmove - resultstress1
    xzeromove = np.zeros((len2,))
    yzero = np.zeros((len2,))
    for i in range(len2 - 1):
        if np.dot(y[i], y[i + 1]) == 0: #% = 0 situation
            if y[i] == 0:
                xzeromove[i] = i
                yzero[i] = 0
            if y[i + 1] == 0:
                xzeromove[i + 1] = i + 1
                yzero[i + 1] = 0
        elif np.dot(y[i], y[i + 1]) < 0: # interpolation
            yzero[i] = np.dot(abs(y[i]) * y_hatmove[-lenf + i] + abs(y[i + 1])*y_hatmove[-lenf - 1 + i], 1/(abs(y[i + 1])+abs(y[i])))
            xzeromove[i] = (yzero[i] - b0_hat) / b1_hat
            a = i
        else:
            pass            
    xzero = resultstrain1[a]
    yzero = resultstress1[a]
    print('Yield point+0.2:', resultstrain1[a],'%')
    print('Yield strength fitted:', resultstress1[a],'MPa')
    

    #Deduce the elestic yield point
    resultstrainlimit = np.array(resultstrain1)
    y_hatlimit = b0_hat + b1_hat*resultstrainlimit
    ylimit = y_hatlimit - resultstress1#first:y_hatmove; second:y_hatf
    xLenlimit = len21
    xzerolimit = np.zeros((xLenlimit,))
    yzerolimit = np.zeros((xLenlimit,))
    nlimit = []
    for i in range(xLenlimit - 1):
        if np.dot(ylimit[i], ylimit[i + 1]) == 0: #% = 0 situation
            if ylimit[i] == 0:
                xzerolimit[i] = i
                yzerolimit[i] = 0
            if ylimit[i + 1] == 0:
                xzerolimit[i + 1] = i + 1
                yzerolimit[i + 1] = 0
        elif np.dot(ylimit[i], ylimit[i + 1]) < 0: # interpolation
            yzerolimit[i] = np.dot(abs(ylimit[i]) * resultstress1[i + 1] + abs(ylimit[i + 1])*resultstress1[i], 1/(abs(ylimit[i + 1])+abs(ylimit[i])))
            nlimit.append(i)
            alimit = np.max(nlimit)
        else:
            pass           
    print('Yield point elastic limit:', resultstrain1[alimit],'%')
    print('Yield elastic limit:', resultstress1[alimit],'MPa')
    
    
    class MyLocator(matplotlib.ticker.AutoMinorLocator):
        def __init__(self, n=2):
            super().__init__(n=n)
    matplotlib.ticker.AutoMinorLocator = MyLocator  
    
    new_ticksx = np.linspace(10,150,15) 
    new_ticksy = np.linspace(0,1400,15)
    figsize = 60, 60
    figure, ax = plt.subplots(figsize = figsize)
    plt.rcParams["xtick.minor.visible"] =  True
    plt.rcParams["ytick.minor.visible"] =  True
    plt.tick_params(labelsize=23, which = 'major', length = 30, width = 6, direction='in', color = 'k')
    plt.tick_params(which = 'minor', length = 25, width = 4, direction='in', color = 'k')

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    plt.xlabel('Strain(%)', fontsize = 140, c = 'k')
    plt.ylabel('Stress(MPa)', fontsize = 140, c = 'k')
    plt.xticks(new_ticksx, c = 'k', fontsize = 100)
    plt.yticks(new_ticksy, c = 'k', fontsize = 100)
    plt.axis([0, dis_x, 0, dis_y])
    bwith = 6
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.plot(resultstrain_plot, resultstress_plot, 'r', linewidth = 14, label = sample_name)
    plt.plot(xdat2, y_hat, 'k--', linewidth = 5)
    plt.scatter(xzero, yzero, color = 'k', marker='o', s = 800)
    plt.scatter(resultstrain1[alimit], resultstress1[alimit], color = 'k', marker='o', s = 800)

    uts_stress = np.max(resultstress_plot)
    uts_strain = resultstrain_plot[np.argmax(resultstress_plot)]
    b0_uts = uts_stress - b1_hat * uts_strain
    uts_x = np.linspace(uts_strain - resultstrain1[alimit] - 20, uts_strain, 10000)
    uts_y = b1_hat * uts_x + b0_uts
    plt.plot(uts_x, uts_y, 'k--', linewidth = 5)
    print(f'UTS strain: {resultstrain_plot[np.argmax(resultstress_plot)]} %')
    print(f'Uniform strain: {-b0_uts / b1_hat} %')

    legend = ax.legend(fontsize = 130, loc = 'upper left', edgecolor='k')
    legend.get_frame().set_facecolor('none')

    plt.show()
    
    end_ = time.time()
    print('calc_cost:', end_ - start_)