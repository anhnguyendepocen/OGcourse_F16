# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:58 2016

@author: Luke
"""

import numpy as np
import Pset_3_func_backup2 as p3
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time


S = [i + 1 for i in range(80)]
#S = [1 ,2 ,3]
nvec = []
for i in S:
    if (i <= round(len(S) * 2 / 3)):
        nvec += [1,]
    else:
        nvec += [0.2,]
nvec = np.array(nvec)
A = 1
alpha = 0.35
delta = 0.05
beta = 0.96
#delta = 0.6415
#beta = 0.442
f_params = np.array([S, nvec, A, alpha, delta, beta])
sigma = 3
L = sum(nvec)
SS_tol =  math.e**(-12)
params = np.array([S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol])

bvec_guess1 = np.ones(len(S) - 1)

bvec_guess2 = \
    np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2])

bvec_guess3 = \
    np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#bvec_guess3 = [0.1, 0.1]
bvec_guess = bvec_guess3
#bvec_guess = np.array([0.01] * (len(S) - 1))
#bvec_guess = np.array([0.01]*(len(S) - 1))



SS_graphs = True
nonSS_graphs = True

maxper = 200



#############
# Problem 1##
#############
var_list = np.array([np.array(['b_2', 'b_3']),np.array(['c_1', 'c_2', 'c_3']),'K'])

f_1 = p3.feasible(params, bvec_guess1)
f_2 = p3.feasible(params, bvec_guess2)
f_3 = p3.feasible(params, bvec_guess3)

#############
# Problem 2##
#############
# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

ss_output= p3.get_SS(params, bvec_guess, SS_graphs)
print("The solution for beta=0.55 is the following")
for i in ss_output:
    print(i,' : ' ,ss_output[i])
plt.title('Figure1: Saving and Consumption Distribution old-age retirement', fontsize=15)
output_path1 = os.path.join(output_dir, "Consumption-Capital SS - old age retirement",)
plt.savefig(output_path1, bbox_inches='tight')
plt.show()

nvec2 = []
for i in S:
    if (i <= round(len(S) * 1 / 2)):
        nvec2 += [1,]
    else:
        nvec2 += [0.2,]
nvec2 = np.array(nvec2)
bvec_guess = np.array([0.01] * (len(S)-1))
params[1] = nvec2

ss_output2= p3.get_SS(params, bvec_guess, SS_graphs)
print("The solution for beta=0.55 is the following")
for i in ss_output:
    print(i,' : ' ,ss_output[i])
plt.title('Figure2 Saving and Consumption Distribution mid-age retirement', fontsize=15)
output_path1 = os.path.join(output_dir, "Consumption-Capital SS - mid age retirement",)
plt.savefig(output_path1, bbox_inches='tight')

bvec_guess = bvec_guess3
params[1] = nvec

# uncomment the follwoing line to see Figure1
# plt.show()


#############
# Problem 3##
#############
if nonSS_graphs:
    b_ss = ss_output['b_ss']
    b_ssvec = np.array([b_ss[13]]*maxper)
    K_ss = ss_output['K_ss']
    kk_start = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])
    K_start = kk_start.sum()
    
    Kpath_lin = p3.get_path(K_start, K_ss, maxper, 'linear')
    Kpath_qdr = p3.get_path(K_start, K_ss, maxper, 'quadratic')
    eql = p3.non_ss (params, bvec_guess, maxper, 'linear')
    Kpath_eql = eql[0]
    w = eql[1]
    r = eql[2]
    bmat = eql[3]
    
    
    age_pers = np.arange(1, maxper+1)
    fig, ax = plt.subplots()
    plt.plot(age_pers, Kpath_lin, marker='D', linestyle=':', label='Linear')
    plt.plot(age_pers, Kpath_qdr, marker='o', linestyle='--', label='Quadratic')
    plt.plot(age_pers, Kpath_eql, marker='x', linestyle='-', label='Equlibrium')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Figure3: Specifications for guess of K time path', fontsize=15)
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Aggregate Cap. Stock $K_t$')
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, "Kpath_init_comp")
    plt.savefig(output_path)
    
    fig, ax = plt.subplots()
    plt.plot(age_pers, w, marker='D', linestyle=':', label='Wage Path')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Figure4: Specifications for Wage Path', fontsize=15)
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Price $K_t$')
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, "W_path")
    plt.savefig(output_path)
    
    fig, ax = plt.subplots()
    plt.plot(age_pers, r, marker='x', linestyle='-', label='Rent Path')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65', linestyle='-')
    plt.title('Figure5: Specifications for Rent Path', fontsize=15)
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Rent $K_t$')
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, "R_path")
    plt.savefig(output_path)
    
    b_15 = b_15 = bmat[13, :]
    b_ssvec = np.array([b_ss[13]]*maxper)
    plt.plot(age_pers, b_15)
    plt.plot(age_pers, b_ssvec)
    plt.xlabel(r'Period $t$')
    plt.ylabel(r'Saving $b_t$')
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, "b15_path")
    plt.savefig(output_path)
