# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 23:31:05 2016

@author: Luke
"""

import numpy as np
import Chap7_ss as p4_ss
import Pset4_nonss as p4_nonss
import elipse_utility as elip
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time
import sys
import os

S = 10
#S = [1 ,2 ,3]
A = 1
alpha = 0.35
delta = 0.05
delta = 1 - (1 - 0.05) ** (80 / S)
beta = 0.96
beta = 0.96 ** (80 / S)
l_endow = 1
chi = 1
EulDiff = True
SS_tol = 1e-14
f_params = (l_endow, A, alpha, delta)
sigma = 3
b_par, miu = elip.elipse(0.8)
SS_graph = True
nonSS_graph = True

SS_params = (S, beta, sigma, A, alpha, delta, chi, b_par, miu, l_endow, EulDiff, SS_tol)

################
##Problem 2#####
################
nvec_guess1 = 0.95 * np.ones(S)
bvec_guess1 = np.ones(S - 1)
f_1 = p4_ss.feasible(f_params, bvec_guess1, nvec_guess1)

nvec_guess2 = 0.95 * np.ones(S)
bvec_guess2 = np.append([0.0], np.ones(S - 2))
f_2 = p4_ss.feasible(f_params, bvec_guess2, nvec_guess2)

nvec_guess3 = 0.95 * np.ones(S)
bvec_guess3 = np.append([0.5], np.ones(S - 2))
f_3 = p4_ss.feasible(f_params, bvec_guess3, nvec_guess3)

nvec_guess4 = 0.5 * np.ones(S)
bvec_guess4 = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
-0.01])
#bvec_guess4 = np.array([-0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.12, 0.1, 0.05, 0.03,
#-0.01])
f_4 = p4_ss.feasible(f_params, bvec_guess4, nvec_guess4)

ss_output = p4_ss.get_SS(SS_params, bvec_guess4, nvec_guess4, SS_graph)
n_ss = ss_output['n_ss']
b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']
L_ss = ss_output['L_ss']

maxiter_TPI = 10000
mindist_TPI = 1e-9
xi = 0.2
TPI_tol = 1e-9
T = 25
weight = []
for i in range(S - 1):
    weight.append((1.5 - 0.87) * i /(S - 2) + 0.87)
weight = np.array(weight)
bvec1 = weight * b_ss

params = (S, T, beta, sigma, chi, b_par, miu, l_endow, A, alpha, delta, b_ss, n_ss, K_ss, C_ss, L_ss, maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff)

non_ss_output = p4_nonss.get_TPI(params, bvec1, nonSS_graph)

n_3 = non_ss_output['npath'][2, :]
b_3 = non_ss_output['bpath'][1, :]

cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Plot time path of aggregate capital stock
tvec = np.linspace(1, T, T)
minorLocator = MultipleLocator(1)
fig, ax = plt.subplots()
plt.plot(tvec, b_3[:T], 'r--', marker='D')
# for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path Saving for Age 3 Agent')
plt.xlabel(r'Period $t$')
plt.ylabel(r'Saving  $K_{t}$')
output_path = os.path.join(output_dir, "Kpath")
plt.savefig(output_path)
plt.show()

# Plot time path of aggregate capital stock
tvec = np.linspace(1, T, T)
minorLocator = MultipleLocator(1)
fig, ax = plt.subplots()
plt.plot(tvec, n_3[:T], marker='D')
# for the minor ticks, use no labels; default NullFormatter
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path Labor Supply for Age 3 Agent')
plt.xlabel(r'Period $t$')
plt.ylabel(r'Labor Supply $K_{t}$')
output_path = os.path.join(output_dir, "Kpath")
plt.savefig(output_path)
plt.show()