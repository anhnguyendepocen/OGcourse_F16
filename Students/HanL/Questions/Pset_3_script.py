# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 14:18:58 2016

@author: Luke
"""

import numpy as np
import Pset_3_func as p3
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time


S = [i + 1 for i in range(80)]
#S = [1 ,2 ,3]
nvec = []
for i in S:
    if (i <= round(len(S) * 2 /3)):
        nvec += [1,]
    else:
        nvec += [0.2]
A = 1
alpha = 0.35
delta = 0.6415
beta = 0.442
f_params = np.array([S, nvec, A, alpha, delta, beta])
sigma = 3
L = np.array(nvec).sum()
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



SS_graphs = True
nonSS_graphs = False

maxper = 50



#############
# Problem 1##
#############
var_list = np.array([np.array(['b_2', 'b_3']),np.array(['c_1', 'c_2', 'c_3']),'K'])

#f_1 = p3.feasible(f_params, bvec_guess1)
#f_2 = p3.feasible(f_params, bvec_guess2)
f_3 = p3.feasible(f_params, bvec_guess3)

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
plt.title('Figure1: Saving and Consumption Distribution beta=0.442', fontsize=15)
output_path1 = os.path.join(output_dir, "Consumption-Capital SS",)
plt.savefig(output_path1, bbox_inches='tight')

# uncomment the follwoing line to see Figure1
# plt.show()
