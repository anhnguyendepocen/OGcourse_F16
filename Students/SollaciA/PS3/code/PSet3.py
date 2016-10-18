# -*- coding: utf-8 -*-
"""
Economic Policy Analysis with Overlapping Genration Models
Problem Set #3
Alexandre B. Sollaci
The Univeristy of Chicago
Fall 2016
"""

import numpy as np
import scipy.optimize as opt
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Define parameters
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

params = (S, beta, sigma, nvec, L, A, alpha, delta, SS_tol)

################ QUESTION 1 #########################

bvec_guess = \
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

def feasible(params, bvec_guess, print_out = True):
    K = sum(bvec_guess) 
    K_cnstr = K <= 0
    w = (1-alpha)*A*(K/L)**(alpha)
    r = alpha*A*(L/K)**(1-alpha) - delta
    b = np.append([0], bvec_guess)
    b = np.append(b, [0])
    c = w*nvec + (1+r)*b[:-1] - b[1:]
    c_cnstr = c <= 0
    b_cnstr = np.zeros(bvec_guess.size, dtype=bool)
    for s in range(1, S-1):
        if c_cnstr[s] == True:
            b_cnstr[s] = True
            b_cnstr[s+1] = True
        b_cnstr[0] = c_cnstr[0]
    if print_out == True:
        string = 'b_cnstr = ' + repr(b_cnstr) + ', c_cnstr = ' + repr(c_cnstr) + \
        ', K_cnstr = ' + repr(K_cnstr)
        print(string)   
    return b_cnstr, c_cnstr, K_cnstr
    
########## QUESTION 2 ###################
    
# guess (feasible) initial values for savings
b_guess = 0.1*np.ones(S-1)

# define SS variables that will be used throughout the question
def SS_vars(b_guess, params):
    S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    K = sum(b_guess)
    Y = A * (K**(alpha)) * (L**(1-alpha))
    r = alpha * A * (L/K)**(1-alpha) - delta
    w = (1-alpha) * A * (K/L)**(alpha)
    b = np.append([0], b_guess)
    b = np.append(b, [0])
    c = w*nvec + (1+r)*b[:-1] - b[1:]
    C = sum(c)
    RCerr = Y - C - delta*K
    EulErr = np.ones(S-1)
    for i in range(S-1):
        EulErr[i] = ( beta*(1+r)*c[i+1]**(-sigma) ) / ( c[i]**(-sigma) ) - 1   
    return K, Y, r, w, c, C, RCerr, EulErr

# Define objective function to feed into solver
def EulerSys(b_guess, *params):
    S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    K, Y, r, w, c, C, RCerr, EulErr = SS_vars(b_guess, params)
    error = c[0:-1]**(-sigma) - beta*(1+r)*c[1:S]**(-sigma)
    return error

# Compute SS values
def get_SS(params, b_guess, SS_graphs = True):
    start_time = time.clock()
    S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    b_ss = opt.fsolve(EulerSys, b_guess, args=(params), xtol = SS_tol)
    K_ss, Y_ss, r_ss, w_ss, c_ss, C_ss, RCerr_ss, EulErr_ss = SS_vars(b_ss, params)
    ss_time = time.clock() - start_time
    ss_output = {'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss, 
                 'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
                 'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
                 'ss_time': ss_time} 
                 
    if SS_graphs == True:
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath("__file__"))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        age = np.arange(1, S+1)
        fig, ax = plt.subplots()
        plt.plot(age, np.append([0], b_ss), marker='D', linestyle=':', label='Savings')
        plt.plot(age, c_ss, marker='o', linestyle='--', label='Consumption')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Distribution of Consumption and Savings by Age', fontsize=15)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'SS Consumption $\bar{c}_{ss}$ and Savings $\bar{b}_{ss}$')
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "cons_savings_dist")
        plt.savefig(output_path)
        plt.show()
    return ss_output   


    
    
    
    
    
    
    
    
    
    
    

