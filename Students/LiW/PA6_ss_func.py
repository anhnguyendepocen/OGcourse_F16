#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 09:55:54 2016

@author: weijiali
"""

import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import PA6 as p
import pandas as pd

### set parameters
S = 80
nvec = 0.2*np.ones(S)
for i in range(int(np.round(2*S/3))):
    nvec[i] = 1.

beta = 0.96
delta = 0.05
sigma = 2.2
A = 1.
alpha = 0.35
g_y = 0.03
SS_tol = 1e-9
EulDiff = True

graphs = True
bvec_guess = 0.1*np.ones(S-1)

fert_rates = p.get_fert(100, False)
mort_rates, infmort_rate = p.get_mort(100, False)
imm_rates = p.get_imm_resid(100, False)

var_names = ('age', 'year1', 'year2')   
pop = pd.read_csv('pop_data.csv', thousands=',', header=0, names=var_names)
pop = pop.as_matrix()

pop1 = pop[:,1]
pop2 = pop[:,2]

g = sum(pop2)/sum(pop1) - 1

Omega = np.diag(imm_rates)
Omega1 = fert_rates*(1 - infmort_rate)
Omega2 = np.diag(1 - mort_rates[:-1]) 
Omega3 = np.vstack((np.zeros([1,99]),Omega2))
Omega4 = np.hstack((Omega3,np.zeros([100,1])))

Omega[0,:] = Omega[0,:] + Omega1
Omega += Omega4

w,v = np.linalg.eig(Omega)

g_bar = w[np.where(w>1)]-1
omega_bar = v[:, np.where(w>1)]
omega_bar = np.reshape(omega_bar,100)

def get_K(barr, omega, imm_rates, g_bar):

    if barr.ndim == 1:  # This is the steady-state case
        K = sum( omega[:-1]*barr + imm_rates[1:]*omega[1:]*barr )  / (1 + g_bar)
        K_cnstr = K <= 0
        if K_cnstr:
            print('get_K() warning: distribution of savings and/or ' +
                  'parameters created K<=0 for some agent(s)')

    elif barr.ndim == 2:  # This is the time path case
        K = barr.sum(axis=0)
        K_cnstr = K <= 0
        if K.min() <= 0:
            print('Aggregate capital constraint is violated K<=0 for ' +
                  'some period in time path.')

    return K, K_cnstr
    
def get_L(nvec, omega):

    L = sum(omega*nvec)

    return L
    

def get_cvec(r, w, bvec, nvec, g_y):

    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    cvec = (1 + r) * b_s + w * nvec - np.exp(g_y)*b_sp1
    if cvec.min() <= 0:
        print('get_cvec() warning: distribution of savings and/or ' +
              'parameters created c<=0 for some agent(s)')
    c_cnstr = cvec <= 0

    return cvec, c_cnstr 

def get_r(params, K, L):
  
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r
    
def get_w(params, K, L):

    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w

    
def get_C(carr, omega):

    if carr.ndim == 1:
        C = sum(omega*carr)
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C
    
def get_Y(params, K, L):

    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y

def feasible(params, bvec):

    nvec, A, alpha, delta, omega, imm_rates, g_bar, g_y = params
    L = get_L(nvec, omega)
    K, K_cnstr = get_K(bvec, omega, imm_rates, g_bar)
    if not K_cnstr:
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        bvec2 = np.append([0], bvec)
        cvec, c_cnstr = get_cvec(r, w, bvec2, nvec, g_y)
        b_cnstr = c_cnstr[:-1] + c_cnstr[1:]

    else:
        c_cnstr = np.ones(cvec.shape[0], dtype=bool)
        b_cnstr = np.ones(cvec.shape[0] - 1, dtype=bool)

    return b_cnstr, c_cnstr, K_cnstr


def get_b_errors(params, r, cvec, c_cnstr, mort_rates, g_y, diff):

    beta, sigma = params
    # Make each negative consumption artifically positive
    cvec[c_cnstr] = 9999.
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    if diff:
        b_errors = (beta * (1 + r) * (1 - mort_rates[:-1]) * np.exp(-sigma*g_y) * mu_cp1) \
                    - mu_c
        b_errors[c_cnstr[:-1]] = 9999.
        b_errors[c_cnstr[1:]] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_cnstr[:-1]] = 9999. / 100
        b_errors[c_cnstr[1:]] = 9999. / 100

    return b_errors


def EulerSys(bvec, *args):

    beta, sigma, nvec, L, A, alpha, delta, g_y, omega, \
        mort_rates, imm_rates, g_bar, EulDiff = args
    K, K_cnstr = get_K(bvec,omega, imm_rates, g_bar)
    if K_cnstr:
        b_err_vec = 1000. * np.ones(nvec.shape[0] - 1)
    else:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        bvec2 = np.append([0], bvec)
        cvec, c_cnstr = get_cvec(r, w, bvec2, nvec, g_y)
        b_err_params = (beta, sigma)
        b_err_vec = get_b_errors(b_err_params, r, cvec, c_cnstr,
                                mort_rates, g_y, EulDiff)

    return b_err_vec


def get_SS(params, bvec_guess, graphs = True):

    start_time = time.clock()
    beta, delta, sigma, A, alpha, nvec, g_y, mort_rates, imm_rates,\
          omega_bar, g_bar, SS_tol, EulDiff = params
    f_params = (nvec, A, alpha, delta, omega_bar, imm_rates, g_bar, g_y)
    b1_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec_guess)
    L = get_L(nvec, omega_bar)
    if K1_cnstr is True or c1_cnstr.max() is True:
        err = ("Initial guess problem: " +
               "One or more constraints not satisfied.")
        print("K1_cnstr: ", K1_cnstr)
        print("c1_cnstr: ", c1_cnstr)
        raise RuntimeError(err)
    else:
        eul_args = (beta, sigma, nvec, L, A, alpha, delta, g_y,\
                    omega_bar, mort_rates, imm_rates, g_bar, EulDiff)
        b_ss = opt.fsolve(EulerSys, bvec_guess, args=(eul_args),
                          xtol=SS_tol)

    # Generate other steady-state values and Euler equations
    K_ss, K_cnstr = get_K(b_ss, omega_bar, imm_rates, g_bar)
    r_params = (A, alpha, delta)
    r_ss = get_r(r_params, K_ss, L)
    w_params = (A, alpha)
    w_ss = get_w(w_params, K_ss, L)
    b_ss2 = np.append([0], b_ss)
    c_ss, c_cnstr = get_cvec(r_ss, w_ss, b_ss2, nvec, g_y)
    Y_params = (A, alpha)
    Y_ss = get_Y(Y_params, K_ss, L)
    C_ss = get_C(c_ss, omega_bar)
    b_err_params = (beta, sigma)
    EulErr_ss = get_b_errors(
        b_err_params, r_ss, c_ss, c_cnstr, mort_rates, g_y, EulDiff)
    RCerr_ss = Y_ss - C_ss - ( (1 + g_bar)*np.exp(g_y) - 1 + delta) * K_ss \
            + np.exp(g_y)*sum(imm_rates[1:]*omega_bar[1:]*b_ss)

    ss_time = time.clock() - start_time

    ss_output = {
        'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
        'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
        'ss_time': ss_time}
    print('b_ss is: ', b_ss)
    print('Euler errors are: ', EulErr_ss)
    print('Resource constraint error is: ', RCerr_ss)

    # Print SS computation time
    print_time(ss_time, 'SS')

    if graphs:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        S           = integer >= 3, number of periods in a life
        age_pers    = (S,) vector, ages from 1 to S
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot steady-state consumption and savings distributions
        S = nvec.shape[0]
        age_pers = np.arange(1, S + 1)
        fig, ax = plt.subplots()
        plt.plot(age_pers, c_ss, marker='D', label='Consumption')
        plt.plot(age_pers, np.hstack((0, b_ss)), marker='D',
                 label='Savings')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-state consumption and savings', fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Units of consumption')
        plt.xlim((0, S + 1))
        plt.ylim((-1.0, 1.15 * (b_ss.max())))
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "SS_bc")
        plt.savefig(output_path)
        # plt.show()

    return ss_output

   
def print_time(seconds, type):
    '''
    --------------------------------------------------------------------
    Takes a total amount of time in seconds and prints it in terms of
    more readable units (days, hours, minutes, seconds)
    --------------------------------------------------------------------
    INPUTS:
    seconds = scalar > 0, total amount of seconds
    type = string, either "SS" or "TPI"
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    OBJECTS CREATED WITHIN FUNCTION:
    secs = scalar > 0, remainder number of seconds
    mins = integer >= 1, remainder number of minutes
    hrs  = integer >= 1, remainder number of hours
    days = integer >= 1, number of days
    FILES CREATED BY THIS FUNCTION: None
    RETURNS: Nothing
    --------------------------------------------------------------------
    '''
    if seconds < 60:  # seconds
        secs = round(seconds, 4)
        print(type + ' computation time: ' + str(secs) + ' sec')
    elif seconds >= 60 and seconds < 3600:  # minutes
        mins = int(seconds / 60)
        secs = round(((seconds / 60) - mins) * 60, 1)
        print(type + ' computation time: ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
    elif seconds >= 3600 and seconds < 86400:  # hours
        hrs = int(seconds / 3600)
        mins = int(((seconds / 3600) - hrs) * 60)
        secs = round(((seconds / 60) - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(hrs) + ' hrs, ' +
              str(mins) + ' min, ' + str(secs) + ' sec')
    elif seconds >= 86400:  # days
        days = int(seconds / 86400)
        hrs = int(((seconds / 86400) - days) * 24)
        mins = int(((seconds / 3600) - days * 24 - hrs) * 60)
        secs = round(
            ((seconds / 60) - days * 24 * 60 - hrs * 60 - mins) * 60, 1)
        print(type + ' computation time: ' + str(days) + ' days, ' +
              str(hrs) + ' hrs, ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
        
        
params = (beta, delta, sigma, A, alpha, nvec, g_y, mort_rates[20:], imm_rates[20:],\
          omega_bar[20:], g_bar, SS_tol, EulDiff)

ss_output = get_SS(params, bvec_guess, True)
