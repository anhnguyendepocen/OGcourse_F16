# script

# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import os
import PS3 as func

S = 80
alpha = 0.35
A = 1
delta = 0.05
beta = 0.96 ** (80/S)
sigma = 3
SS_tol = 1e-13

# nvec
nvec = np.zeros(S)
nvec[:int(round(2 * S / 3))] = 1
nvec[int(round(2 * S / 3)):] = 0.2

# L
L = nvec.sum()

params = (S, beta, sigma, L, A, alpha, delta, SS_tol)



# Question 1: Feasibility
# (a)
bvec_guess1 = np.ones(S-1)
print("(a) : ", func.feasible(S, alpha, A, delta, bvec_guess1))

# (b)
bvec_guess2 = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2])
print("(b) : ", func.feasible(S, alpha, A, delta, bvec_guess2))

# (c)
bvec_guess3 = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print("(c) : ", func.feasible(S, alpha, A, delta, bvec_guess3))


# Question 2: Solve SS equilibrium
#(a)
SS = func.get_SS(params, bvec_guess3, nvec, False)
print("ss outputs : ", SS)

#(b) retire early
nvec2 = np.zeros(S)
nvec2[:int(round(S / 2))] = 1
nvec2[int(round(S / 2)):] = 0.2

L2 = nvec2.sum()

params = (S, beta, sigma, L2, A, alpha, delta, SS_tol)
SS2 = func.get_SS(params, bvec_guess3, nvec2, False)
print ("ss retire early : ", SS2)








