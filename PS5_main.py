import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

import PS5_ssfunc as ssf
import PS5_TPI as tpi


##################Question 2 #######################
S = 80
nvec = 0.2*np.ones(S)
for i in range(int(np.round(2*S/3))):
    nvec[i] = 1.

beta = 0.96
sigma = 2.2
chi_b = 1.
zeta_s = 1/S
L = sum(nvec)
A = 1.
alpha = 0.35
delta = 0.05
SS_tol = 1e-9


bvec_guess = 0.1 * np.ones(S) 

#L = ssf.get_L(nvec)
#K, K_cnstr = ssf.get_K(bvec_guess)
#w_params = (A, alpha)
#w = ssf.get_w(w_params, K, L)
#r_params = (A, alpha, delta)
#r = ssf.get_r(r_params, K, L)
# 
SS_params = beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta, SS_tol
#
ss_output = ssf.SS(SS_params, bvec_guess, True)



############## QUESTION 3 ######################

b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']
maxiter_TPI = 400
mindist_TPI = 1e-9
xi = 0.8
T = 300
TPI_tol = 1e-9

bvec1 = 0.8*b_ss

TPI_params = (S, T, beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss,
        maxiter_TPI, mindist_TPI, xi, TPI_tol)

tpi_output = tpi.get_TPI(TPI_params, bvec1, True) 
