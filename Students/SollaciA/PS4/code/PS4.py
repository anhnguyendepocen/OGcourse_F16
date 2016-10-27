# -*- coding: utf-8 -*-
"""
Economic Policy Analysis with Overlapping Genration Models
Problem Set #4
Alexandre B. Sollaci
The Univeristy of Chicago
Fall 2016
"""

import numpy as np
import scipy.optimize as opt
#import time
import functions_question_2 as fun2
import functions_question_3 as fun3
import functions_question_4 as fun4
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

############################# QUESTION 1 #########################

theta = 0.8
nvec = np.linspace(0.05, 0.95, 1000)

def dist_utility(params_guess, nvec):
    b, mu = params_guess
    v_cfe = -nvec**(1/theta)
    v_elp = -b*(1-nvec**mu)**(1/mu - 1) * nvec**(mu-1)
    SQE = sum( (v_cfe - v_elp)**2 )
    return SQE

params_guess = np.array([1, 1])
bnds = ((0, None), (0, None))
res = opt.minimize(dist_utility, params_guess, args=(nvec),
                   method='L-BFGS-B',bounds = bnds, options={'disp': True, 'ftol': 1e-8})
SSE = res.fun
b, mu = res.x

lvec = 1 - nvec
mv_cfe = (1-lvec)**(1/theta)
mv_elp = b*(1 - (1 - lvec)**(mu))**(1/mu - 1) * (1-lvec)**(mu-1)

# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath("__file__"))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


fig, ax = plt.subplots()
plt.plot(lvec, mv_cfe, label='CFE')
plt.plot(lvec, mv_elp, label='Elliptical')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Different Specifications for Marginal Utility', fontsize=20)
plt.xlabel(r'Leisure $l$')
plt.ylabel(r'Marginal Utility')
#plt.xlim((0, S + 1))
#plt.ylim((-1.0, 1.15 * (b_ss.max())))
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "utility_leisure")
plt.savefig(output_path)
# plt.show()

########################### QUESTION 2 ####################################

S = 10
A = 1.
alpha = 0.35
beta = 0.96**(80/S)
delta = 1 - (1 - 0.05)**(80/S)
sigma = 3
l_tilde = 1.
SS_tol = 1e-13
EulDiff = True

nvec_guess = 0.95*np.ones(S)
bvec_guess = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1, -0.01])

f_params = (l_tilde, A, alpha, delta)

n_low, n_high, b_cnstr, c_cnstr, K_cnstr = fun2.feasible(f_params, nvec_guess, bvec_guess)

######################## QUESTION 3 ######################################

# initial guesses for interest rate and wage
r = .4
w = .4

omega = 0.8
tol = 1e-9
r_error = 10
w_error = 10
max_iter = 1000
iter = 1


while (r_error > tol or w_error > tol) and iter < max_iter:
    params = (S, beta, sigma, b, mu, l_tilde, A, alpha, delta, SS_tol, EulDiff, r, w)
    
    ss_output = fun3.get_SS(params, bvec_guess, nvec_guess, graphs = False)
    
    w_ss = ss_output['w_ss']
    r_ss = ss_output['r_ss']
    b_ss = ss_output['b_ss']
    n_ss = ss_output['n_ss']

    r_error = abs(r - r_ss)
    w_error = abs(w - w_ss)
    if r_error > tol or w_error > tol:
        r = omega*r + (1-omega)*r_ss
        w = omega*w + (1-omega)*w_ss
    iter += 1

params = (S, beta, sigma, b, mu, l_tilde, A, alpha, delta, SS_tol, EulDiff, r_ss, w_ss)
ss_output = fun3.get_SS(params, b_ss, n_ss, graphs = True)

error = ss_output['EulErr_ss']
max_error = max( max(error) , max(-error) )
print("The maximum Euler error is: ", max_error)

############################# QUESTION 4 ###################################

x = np.zeros(S+1)
for i in range(S+1):
    x[i] = (1.5 - 0.87)/(S-2) * (i - 2) + 0.87
x = x[2:]
b_init = b_ss*x

T = 50
L = sum(n_ss)

b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']
maxiter_TPI = 100
mindist_TPI = 1e-9
xi = 0.8
bvec1 = b_init
nvec1 = n_ss

TPI_tol = 1e-9
tpi_params = (S, T, beta, sigma, A, alpha, delta, b, mu, l_tilde, b_ss, K_ss, C_ss,
        maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff)

#tpi_output = fun4.get_TPI(tpi_params, bvec1, nvec1, graphs = True)



