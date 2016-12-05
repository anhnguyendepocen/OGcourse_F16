# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 00:42:13 2016

@author: Luke
"""

import pandas as pd
import os
import numpy as np
import weighted
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
#import weighted as wgted

cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "data"
output_dir = os.path.join(cur_path, output_fldr)

os.chdir(output_dir)

df_main = pd.read_stata('p13i6.dta', columns = ['X5804', 'X5805', 'X5809', 'X5810', 'X5814', 'X5815', 'X8022'])
df_summ = pd.read_stata('rscfp2013.dta', columns = ['networth', 'age', 'wgt'])
#print(df_main[df_main.X8022 >= 21].X5809.describe())

######################
###Problem 1 #########
######################
#adjust for GDP
def problem1(df_main, df_summ):
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    df_main['X5804'][df_main.X5805 == 2012] = df_main['X5804'][df_main.X5805 == 2012] * 0.9854
    df_main['X5804'][df_main.X5805 == 2011] = df_main['X5804'][df_main.X5805 == 2011] * 0.9652
    
    df_main['X5809'][df_main.X5810 == 2012] = df_main['X5809'][df_main.X5810 == 2012] * 0.9854
    df_main['X5809'][df_main.X5810 == 2011] = df_main['X5809'][df_main.X5810 == 2011] * 0.9652
    
    df_main['X5814'][df_main.X5815 == 2012] = df_main['X5814'][df_main.X5815 == 2012] * 0.9854
    df_main['X5814'][df_main.X5815 == 2011] = df_main['X5814'][df_main.X5815 == 2011] * 0.9652
    
    df_main_age = df_main[df_main.X8022 >= 21]
    df_summ_age = df_summ[df_summ.age >= 21]
    
    #df_main_age.loc[:, ['X5804','X5805', 'X5809','X5810', 'X5814','X5815' ]]
    #cri1 = df_main_age['X5805'] == 0
    #cri11 = min(df_main_age['X5805'] > 2013, df_main_age['X5805'] < 2011)
    cri1 = df_main_age['X5805'] >= 2011
    cri2 = df_main_age['X5810'] >= 2011
    cri3 = df_main_age['X5815'] >= 2011
    
    
#    t_bequest = {}
    df_main_age['year1'] = 0
    df_main_age.loc[:, 'year1'][cri1] = df_main_age.loc[:, 'X8022'][cri1] - (2013 - df_main_age.loc[:, 'X5805'][cri1])
    df_main_age.loc[:, 'year1'][cri2] = df_main_age.loc[:, 'X8022'][cri2] - (2013 - df_main_age.loc[:, 'X5810'][cri2])
    df_main_age.loc[:, 'year1'][cri3] = df_main_age.loc[:, 'X8022'][cri3] - (2013 - df_main_age.loc[:, 'X5815'][cri3])
    df_main_age['bequest'] = 0
    df_main_age.loc[:, 'bequest'][cri1] = df_main_age.loc[:, 'bequest'][cri1] + df_main_age.loc[:, 'X5804'][cri1] 
    df_main_age.loc[:, 'bequest'][cri2] = df_main_age.loc[:, 'bequest'][cri2] + df_main_age.loc[:, 'X5809'][cri2]
    df_main_age.loc[:, 'bequest'][cri3] = df_main_age.loc[:, 'bequest'][cri3] + df_main_age.loc[:, 'X5814'][cri3] 
    #df_main_age['year1'][cri1] = 0
    #df_main_age['bequest_1'] = df_main_age['X5804']
    #df_main_age['bequest_1'][cri1] = 0
    #df_main_age['year1'][df_main_age.year1 < 21] = 0
    #df_main_age['bequest'][df_main_age.year1 < 21] = 0
    df_merge = pd.merge(df_main_age, df_summ_age, left_index = True, right_index = True)
    
    bq_dict = {'year': np.arange(21, 101, 1), 'bequest':0}
    df_bequest = pd.DataFrame(bq_dict)
    #for i, row in df_merge.iterrows():
    #    if row['bequest'] != 0 and row['year1'] != 0:
    #        t_bequest[int(row['year1'])] = t_bequest.get(int(row['year1']), 0) + row['bequest'] * row['wgt']
    for i, row in df_merge.iterrows():
        if row['bequest'] != 0 and row['year1'] != 0:
            df_bequest.loc[:, 'bequest'][df_bequest.year == row['year1']] += row['bequest'] * row['wgt']
    
    #total = sum(list(t_bequest.values()))  
    total = df_bequest.bequest.sum()  
    
    df_bequest.bequest = df_bequest.bequest / total              
    #bequest_dict = {}
    #for key, value in t_bequest.items():
    #    val_year = bequest_dict.get('year', np.array([], dtype = 'int64'))
    #    val_beq = bequest_dict.get('bequest', np.array([]))
    #    bequest_dict['year'] = np.append(val_year, int(key))
    #    bequest_dict['bequest'] = np.append(val_beq, (value / total) * 100)
    #df_bequest = pd.DataFrame(bequest_dict)
    df_bequest = df_bequest.sort_values('year')
    df_bequest.plot(x = 'year', y = 'bequest', kind = 'line', xlim = (20, 101))
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Bequest Distribution by Age', fontsize=20)
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'Bequest Percent')
    output_path = os.path.join(output_dir, "dist_bq_age")
    plt.savefig(output_path)
    return df_bequest
    
df_bequest = problem1(df_main, df_summ)

def problem2(df_main, df_summ):
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    df_main['X5804'][df_main.X5805 == 2012] = df_main['X5804'][df_main.X5805 == 2012] * 0.9854
    df_main['X5804'][df_main.X5805 == 2011] = df_main['X5804'][df_main.X5805 == 2011] * 0.9652
    
    df_main['X5809'][df_main.X5810 == 2012] = df_main['X5809'][df_main.X5810 == 2012] * 0.9854
    df_main['X5809'][df_main.X5810 == 2011] = df_main['X5809'][df_main.X5810 == 2011] * 0.9652
    
    df_main['X5814'][df_main.X5815 == 2012] = df_main['X5814'][df_main.X5815 == 2012] * 0.9854
    df_main['X5814'][df_main.X5815 == 2011] = df_main['X5814'][df_main.X5815 == 2011] * 0.9652
    
    df_main_age = df_main[df_main.X8022 >= 21]
    df_summ_age = df_summ[df_summ.age >= 21]
    
    #df_main_age.loc[:, ['X5804','X5805', 'X5809','X5810', 'X5814','X5815' ]]
    #cri1 = df_main_age['X5805'] == 0
    #cri11 = min(df_main_age['X5805'] > 2013, df_main_age['X5805'] < 2011)
    cri1 = df_main_age['X5805'] >= 2011
    cri2 = df_main_age['X5810'] >= 2011
    cri3 = df_main_age['X5815'] >= 2011
    
    
#    t_bequest = {}
    df_main_age['year1'] = 0
    df_main_age.loc[:, 'year1'][cri1] = df_main_age.loc[:, 'X8022'][cri1] - (2013 - df_main_age.loc[:, 'X5805'][cri1])
    df_main_age.loc[:, 'year1'][cri2] = df_main_age.loc[:, 'X8022'][cri2] - (2013 - df_main_age.loc[:, 'X5810'][cri2])
    df_main_age.loc[:, 'year1'][cri3] = df_main_age.loc[:, 'X8022'][cri3] - (2013 - df_main_age.loc[:, 'X5815'][cri3])
    df_main_age['bequest'] = 0
    df_main_age.loc[:, 'bequest'][cri1] = df_main_age.loc[:, 'bequest'][cri1] + df_main_age.loc[:, 'X5804'][cri1] 
    df_main_age.loc[:, 'bequest'][cri2] = df_main_age.loc[:, 'bequest'][cri2] + df_main_age.loc[:, 'X5809'][cri2]
    df_main_age.loc[:, 'bequest'][cri3] = df_main_age.loc[:, 'bequest'][cri3] + df_main_age.loc[:, 'X5814'][cri3] 
    #df_main_age['year1'][cri1] = 0
    #df_main_age['bequest_1'] = df_main_age['X5804']
    #df_main_age['bequest_1'][cri1] = 0
    #df_main_age['year1'][df_main_age.year1 < 21] = 0
    #df_main_age['bequest'][df_main_age.year1 < 21] = 0
    df_merge = pd.merge(df_main_age, df_summ_age, left_index = True, right_index = True)
    cut_off1 = weighted.quantile_1D(df_merge.networth, df_merge.wgt, 0.25)
    cut_off2 = weighted.quantile_1D(df_merge.networth, df_merge.wgt, 0.5)
    cut_off3 = weighted.quantile_1D(df_merge.networth, df_merge.wgt, 0.75)
    df_merge.loc[:, 'ability'] = 0
    df_merge.ability[df_merge.networth <= cut_off1] = 1
    df_merge.ability[(df_merge.networth <= cut_off2) & (df_merge.networth > cut_off1)] =2
    df_merge.ability[(df_merge.networth <= cut_off3) & (df_merge.networth > cut_off2)] =3
    df_merge.ability[df_merge.networth > cut_off3] = 4

    bq_dict = {'year': np.array(list(range(21, 101)) * 4), 'bequest':0, \
               'ability': np.array([1] * 80 + [2] * 80 + [3] * 80 + [4] * 80)}
    df_bequest = pd.DataFrame(bq_dict)
    #for i, row in df_merge.iterrows():
    #    if row['bequest'] != 0 and row['year1'] != 0:
    #        t_bequest[int(row['year1'])] = t_bequest.get(int(row['year1']), 0) + row['bequest'] * row['wgt']
    for i, row in df_merge.iterrows():
        if row['bequest'] != 0 and row['year1'] != 0:
            df_bequest.loc[:, 'bequest'][(df_bequest.year == row['year1']) & (df_bequest.ability == row.ability)] \
                        += row['bequest'] * row['wgt']
    
    #total = sum(list(t_bequest.values()))  
    total = df_bequest.bequest.sum()  
    
    df_bequest.bequest = df_bequest.bequest / total              
    #bequest_dict = {}
    #for key, value in t_bequest.items():
    #    val_year = bequest_dict.get('year', np.array([], dtype = 'int64'))
    #    val_beq = bequest_dict.get('bequest', np.array([]))
    #    bequest_dict['year'] = np.append(val_year, int(key))
    #    bequest_dict['bequest'] = np.append(val_beq, (value / total) * 100)
    #df_bequest = pd.DataFrame(bequest_dict)
#    df_bequest = df_bequest.sort_values('year')
#    df_bequest.plot(x = 'year', y = 'bequest', kind = 'line', xlim = (20, 101))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax = fig.gca(projection='3d')
    x = np.array(df_bequest.year)
    y = np.array(df_bequest.ability)
    z = np.array(df_bequest.bequest)
#    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
    surf = ax.plot_trisurf(df_bequest.year, df_bequest.ability, df_bequest.bequest,\
                            cmap=cm.coolwarm, linewidth=0.1)
#    ax.set_zlim(-1.01, 1.01)
    
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Bequest by age and ability group')
    ax.set_xlabel('Age')
    ax.set_ylabel('Net Worth Group')
    ax.set_zlabel('Bequest Percent')
    output_path = os.path.join(output_dir, "dist_bq_age_ab")
    plt.savefig(output_path)
    plt.show()
    
    return df_bequest

df_bequest2 = problem2(df_main, df_summ)
