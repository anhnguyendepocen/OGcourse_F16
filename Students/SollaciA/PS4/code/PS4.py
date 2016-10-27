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
b_ellip, mu = res.x

lvec = 1 - nvec
mv_cfe = (1-lvec)**(1/theta)
mv_elp = b_ellip*(1 - (1 - lvec)**(mu))**(1/mu - 1) * (1-lvec)**(mu-1)

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
chi_n_vec = 1.
SS_tol = 1e-13
EulDiff = True

nvec_guess = 0.95*np.ones(S)
bvec_guess = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1, -0.01])

f_params = (l_tilde, A, alpha, delta)

n_low, n_high, b_cnstr, c_cnstr, K_cnstr = fun2.feasible(f_params, nvec_guess, bvec_guess)

######################## QUESTION 3 ######################################

params = (S, beta, sigma, A, alpha, delta, chi_n_vec, b_ellip, mu, l_tilde, EulDiff, SS_tol)
ss_output = fun3.get_SS(params, bvec_guess, nvec_guess, graphs = True)

error = ss_output['EulErr_ss']
max_error = max( max(error) , max(-error) )
print("The maximum Euler error is: ", max_error)

############################# QUESTION 4 ###################################

b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']
L_ss = ss_output['L_ss']
n_ss = ss_output['n_ss']


x = np.zeros(S+1)
for i in range(S+1):
    x[i] = (1.5 - 0.87)/(S-2) * (i - 2) + 0.87
x = x[2:]
b_init = b_ss*x

T = 50
maxiter_TPI = 5000
mindist_TPI = 1e-9
xi = 0.8
TPI_tol = 1e-9

tpi_params = (S, T, beta, sigma, chi_n_vec, b_ellip, mu, l_tilde, A, alpha, 
          delta, b_ss, n_ss, K_ss, C_ss, L_ss, maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff)

tpi_output = fun4.get_TPI(tpi_params, b_init, True)


# Plots
npath3 = tpi_output['npath'][2, :]
bpath3 = tpi_output['bpath'][1, :]

cur_path = os.path.split(os.path.abspath("__file__"))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Plot time path of aggregate capital stock
period = np.linspace(1, T, T)
minorLocator = MultipleLocator(1)
fig, ax = plt.subplots()
plt.plot(period, bpath3[:T], marker='D')
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path savings for 3-period olds')
plt.xlabel(r'Period $t$')
plt.ylabel(r'Saving$')
output_path = os.path.join(output_dir, "tpi_s")
plt.savefig(output_path)
plt.show()


minorLocator = MultipleLocator(1)
fig, ax = plt.subplots()
plt.plot(period, npath3[:T], marker='D')
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Time path labor supply for 3-period olds')
plt.xlabel(r'Period $t$')
plt.ylabel(r'Labor Supply')
output_path = os.path.join(output_dir, "tpi_n")
plt.savefig(output_path)
plt.show()







