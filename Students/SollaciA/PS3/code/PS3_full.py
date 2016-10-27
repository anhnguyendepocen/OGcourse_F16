# -*- coding: utf-8 -*-
"""
Economic Policy Analysis with Overlapping Genration Models
Problem Set #3
Alexandre B. Sollaci
The Univeristy of Chicago
Fall 2016
"""

import numpy as np
#import scipy.optimize as opt
#import time
import functions_PS3 as functions
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
#import os


S = 80
nvec = 0.2*np.ones(S)
for i in range(int(np.round(2*S/3))):
    nvec[i] = 1.
A = 1.
alpha = 0.35
delta = 0.05
beta = 0.96
sigma = 3
L = sum(nvec)
SS_tol = 1e-13
EulDiff = True

params = (nvec, A, alpha, delta)

bvec_guess = 0.1*np.ones(S-1)

b_cnstr, c_cnstr, K_cnstr = functions.feasible(params, bvec_guess)

params = (beta, sigma, nvec, L, A, alpha, delta, SS_tol, EulDiff)

ss_output = functions.get_SS(params, bvec_guess, graphs = True)

b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']
maxiter_TPI = 400
mindist_TPI = 1e-9
xi = 0.8
bvec1 = 0.8*b_ss
T = 320
TPI_tol = 1e-9
tpi_params = (S, T, beta, sigma, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss,
        maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff)

tpi_output = functions.get_TPI(tpi_params, bvec1, graphs = True)



