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
nvec[]
for i in len(S):
    if (i <= round(len(S) * 2 /3)):
        nvec.append(1)
    else:
        nvec.append(0.2)
A = 1
alpha = 0.35
delta = 0.6415
beta = 0.442
f_params = np.array([S, nvec, A, alpha, delta, beta])
sigma = 3
L = nvec.sum()
SS_tol =  math.e**(-12)
nvec=np.array([1, 1, 0.2])
params = np.array([S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol])

bvec_guess1 = np.ones(len(S) - 1)
#bvec_guess2 = np.array([0.06, -0.001])
##bvec_guess3 = np.array([0.1, 0.1])
#bvec_guess = bvec_guess3

SS_graphs = False
nonSS_graphs = False

maxper = 50



#############
# Problem 1##
#############
var_list = np.array([np.array(['b_2', 'b_3']),np.array(['c_1', 'c_2', 'c_3']),'K'])
f_1 = p3.feasible(f_params, bvec_guess1)
#f_2 = p3.feasible(f_params, bvec_guess2)
#f_3 = p3.feasible(f_params, bvec_guess3)