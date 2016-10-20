# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:57:06 2016

@author: Alexandre
"""

import numpy as np
import scipy.optimize as opt
import time

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# Question 1

S = 3
nvec = np.array([1., 1., 0.2])
A = 1.
alpha = 0.35
delta = 0.6415
beta = 0.442
sigma = 3
L = sum(nvec)
SS_tol = 1e-13

params = (S, beta, sigma, nvec, L, A, alpha, delta, SS_tol)
bvec_guess = np.array([.1, .1])

def feasible(params, bvec_guess, print_out = True):
    K_cnstr = sum(bvec_guess) <= 0 
    w = (1-alpha)*A*(sum(bvec_guess)/2)**(alpha)
    r = alpha*A*(2/sum(bvec_guess))**(1-alpha) - delta
    b = np.append([0], bvec_guess)
    c = np.zeros(3)
    for s in range(S):
        if s <= 1:
            c[s] = w*nvec[s] + (1 + r)*b[s] - b[s+1]
        else:
            c[s] = w*nvec[s] + (1 + r)*b[s]
    c_cnstr = c <= 0
    if c_cnstr[1] == True:
        b_cnstr = np.array([True, True], dtype=bool)
    else:
        b_cnstr = np.array([c_cnstr[0], c_cnstr[2]], dtype=bool)       
    if print_out == True:
        string = 'b_cnstr = ' + repr(b_cnstr) + ', c_cnstr = ' + repr(c_cnstr) + \
        ', K_cnstr = ' + repr(K_cnstr)
        print(string)   
    return b_cnstr, c_cnstr, K_cnstr

# Question 2

def SS_vars(bvec_guess, params):
    S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    K = sum(bvec_guess)
    Y = A * (K**(alpha)) * (L**(1-alpha))
    r = alpha * A * (L/K)**(1-alpha) - delta
    w = (1-alpha) * A * (K/L)**(alpha)
    b = np.append([0], bvec_guess)
    b = np.append(b, [0])
    c = np.zeros(S)
    for i in range(len(c)):
        c[i] = w*nvec[i] + (1+r)*b[i] - b[i+1]
    C = sum(c)
    RCerr = Y - C - delta*K
    EulErr = np.ones(S-1)
    for i in range(S-1):
        EulErr[i] = ( beta*(1+r)*c[i+1]**(-sigma) ) / ( c[i]**(-sigma) ) - 1   
    return K, Y, r, w, c, C, RCerr, EulErr


def SS_eqns(bvec_guess, *params):
    S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    K, Y, r, w, c, C, RCerr, EulErr = SS_vars(bvec_guess, params)
    obj = np.zeros(S-1)
    for i in range(S-1):
        obj[i] = c[i]**(-sigma) - beta*(1+r)*c[i+1]**(-sigma)
    return obj

def get_SS(params, bvec_guess, SS_graphs = False):
    start_time = time.clock()
    S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
    b_ss = opt.fsolve(SS_eqns, bvec_guess, args=(params), xtol = SS_tol)
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
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'SS Consumption $\bar{c}_{ss}$ and Savings $\bar{b}_{ss}$')
        plt.legend(loc='right')
        output_path = os.path.join(output_dir, "cons_savings_dist")
        plt.savefig(output_path)
        plt.show()
    return ss_output


    
# Question 3

T = 50
epsilon = 1e-9
xi = 0.8

dict = get_SS(params, bvec_guess)
b_ss = dict['b_ss']
K_ss = dict['K_ss']
b_init =  np.array([0.8*b_ss[0], 1.1*b_ss[1]])
K_init = sum(b_init)
    
# time path
K = np.linspace(K_init, K_ss, T)
r = np.zeros(T)
w = np.zeros(T)
error = 10

while (error > epsilon):
    for t in range(T):
        r[t] = alpha * A * (L/K[t])**(1-alpha) - delta
        w[t] = (1-alpha) * A * (K[t]/L)**(alpha)
    
    tpi_vars = (sigma, beta, r, w, t)
    # compute b_32
    def b_32_eqn(b, *tpi_vars):
        sigma, beta, r, w, t = tpi_vars
        obj = ( w[0] + (1 + r[0])*b_init[0] - b )**(-sigma) - \
               beta*(1 + r[1])*( (1 + r[1])*b + 0.2*w[1] )**(-sigma)
        return obj
    
    b_init_guess = sum(b_init)/2
    b_32 = opt.fsolve(b_32_eqn, b_init_guess, args=(tpi_vars), xtol = epsilon)
    
    # get rest of savings b[0] = b_2,t+1; b[1] = b_3,t+2
    def b_eqns(b, *tpi_eul_vars):
        beta, sigma, r, w, t = tpi_eul_vars
        obj = np.zeros(len(b))
        obj[0] = ( w[t] - b[0])**(-sigma) - \
            beta*(1 + r[t+1])*(w[t+1] + (1 + r[t+1])*b[0] - b[1] )**(-sigma)
        obj[1] = (w[t+1] + (1 + r[t+1])*b[0] - b[1]  )**(-sigma) - \
            beta*(1 + r[t+2])*( (1 + r[t+2]) * b[1] + 0.2*w[t+2] )**(-sigma)
        return obj
    
    b2 = np.zeros(T-1)
    b3 = np.zeros(T-1)
    b_guess = b_init
    for t in range(T-2): 
        tpi_eul_vars = (beta, sigma, r, w, t)
        x, y = opt.fsolve(b_eqns, b_guess, args=(tpi_eul_vars), xtol = epsilon)
        b_guess = np.array([x, y])
        b2[t] = x
        b3[t+1] = y
    
    # construct the savings time path
    b2[T-2] = b_ss[0]
    b2 = np.append(b_init[0], b2)
    b3[0] = b_32
    b3 = np.append(b_init[1], b3)
    b_init = np.array([b2[0], b3[0]])
    
    # update K
    K_prime = b2 + b3
    error = np.linalg.norm(K_prime - K)
    if error > epsilon:
        K = xi*K + (1-xi)*K_prime

# Find how long it takes for convergence

conv_period = np.where(abs(K - K_ss) < 0.0001)
# all indices where abs(K - K_ss) < 0.0001

# Plot K and initial K guess

K_plot = np.append(K, [K_ss, K_ss, K_ss, K_ss, K_ss])
K_linear = np.append(np.linspace(K_init, K_ss, T), [K_ss, K_ss, K_ss, K_ss, K_ss] )

cur_path = os.path.split(os.path.abspath("__file__"))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

period = np.arange(T+5)
fig, ax = plt.subplots()
plt.plot(period, K_linear , marker='D', linestyle=':', label='Linear Guess')
plt.plot(period, K_plot, marker='o', linestyle='--', label='Time Path')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Capital Time Path to Steady State', fontsize=15)
plt.xlabel(r'Period $t$')
plt.ylabel(r'Capital')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "time_path_K")
plt.savefig(output_path)
plt.show()

# Plot w and r 

w_ss = dict['w_ss']
r_ss = dict['r_ss']

w_plot = np.append(w, [w_ss, w_ss, w_ss, w_ss, w_ss])
r_plot = np.append(r, [r_ss, r_ss, r_ss, r_ss, r_ss])

period = np.arange(T+5)
fig, ax = plt.subplots()
plt.plot(period, w_plot , marker='D', linestyle=':', label='Wage')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Wage Time Path to Steady State', fontsize=15)
plt.xlabel(r'Period $t$')
plt.ylabel(r'$w$')
#plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "time_path_w")
plt.savefig(output_path)
plt.show()

period = np.arange(T+5)
fig, ax = plt.subplots()
plt.plot(period, r_plot , marker='D', linestyle=':', label='Interest Rate')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Interest Rate Time Path to Steady State', fontsize=15)
plt.xlabel(r'Period $t$')
plt.ylabel(r'$r$')
#plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "time_path_r")
plt.savefig(output_path)
plt.show()
