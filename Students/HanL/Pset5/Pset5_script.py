#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 02:00:46 2016

@author: luxihan
"""

import numpy as np
import Pset5_ss as p5_ss
import Pset5_nonss as p5_nonss
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time
import pandas as pd
import pset5_p1 as p5p1

cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "data"
output_dir = os.path.join(cur_path, output_fldr)

os.chdir(output_dir)

df_main = pd.read_stata('p13i6.dta', columns = ['X5804', 'X5805', 'X5809', 'X5810', 'X5814', 'X5815', 'X8022'])
df_summ = pd.read_stata('rscfp2013.dta', columns = ['networth', 'age', 'wgt'])

df_bequest = p5p1.problem1(df_main, df_summ)
zeta_s1 = np.array(df_bequest.bequest)


S = 80
#S = [1 ,2 ,3]
nvec = []
for i in range(S):
    if (i <= round(S * 2 / 3)):
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
zeta_s2 = np.array([1 / S] * S)
sigma = 2.2
chib = 1
L = sum(nvec)
SS_tol =  math.e**(-20)
EulDiff = True
params = (beta, sigma, chib, zeta_s1, nvec, L, A, alpha, delta, SS_tol, EulDiff)
f_params = (nvec, A, alpha, delta, zeta_s1)
SS_graph = True
nonSS_graph = True

bvec_guess1 = np.ones(S)

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
    -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.2])

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
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#bvec_guess3 = [0.1, 0.1]
bvec_guess = bvec_guess3
f_3 = p5_ss.feasible(f_params, bvec_guess)
ss_output = p5_ss.get_SS(params, bvec_guess, SS_graph)


ss_output = p5_ss.get_SS(params, bvec_guess, SS_graph)
b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']

maxiter_TPI = 900
mindist_TPI = 1e-9
xi = 0.7
TPI_tol = 1e-9
T = 320
params      = (S, T, beta, sigma, chib, zeta_s1, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss, maxiter_TPI, \
            mindist_TPI, xi, TPI_tol, EulDiff)
weight = []
for i in range(S):
    weight.append((1.5 - 0.87) * i /(S - 1) + 0.87)
weight = np.array(weight)
bvec1 = weight * b_ss
non_ss_output = p5_nonss.get_TPI(params, bvec1, nonSS_graph)
plt.show()

age_pers = np.arange(0, T + 1, 1)
b_path25 = non_ss_output['bpath'][25 - 2, : T + 1]
plt.plot(age_pers, b_path25)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path for saving of Age 25 Agents')
plt.xlabel(r'Period $t$')
plt.ylabel(r'Saving $b_{t}$')
output_path = os.path.join(cur_path, 'images/saving_25')
plt.savefig(output_path)
plt.show()
