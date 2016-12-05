#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:45:47 2016

@author: luxihan
"""

# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

def get_r(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state interest rate r or time path of interest
    rates r_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 3 tuple, (A, alpha, delta)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    r = scalar > 0 or (T+S-2) vector, steady-state interest rate or time
        path of interest rate

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: r
    --------------------------------------------------------------------
    '''
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r
    
def get_w(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state wage w or time path of wages w_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 2 tuple, (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    w = scalar > 0 or (T+S-2) vector, steady-state wage or time path of
        wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: w
    --------------------------------------------------------------------
    '''
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w
    

def get_K(barr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate capital stock K or time path of
    aggregate capital stock K_t
    --------------------------------------------------------------------
    INPUTS:
    barr = (S,) vector or (S, T+S-2) matrix, values for steady-state
           savings (b_2,b_3,...b_S, b_S+1) or time path of distribution of
           savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    K       = scalar or (T+S-2,) vector, steady-state aggregate capital
              stock or time path of aggregate capital stock
    K_cnstr = Boolean or (T+S-2) Boolean, =True if K <= 0 or if K_t <= 0

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, K_cnstr
    --------------------------------------------------------------------
    '''
    if barr.ndim == 1:  # This is the steady-state case
        K = barr.sum()
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
    
    
def get_L(nvec):
    '''
    --------------------------------------------------------------------
    Solve for aggregate labor L
    --------------------------------------------------------------------
    INPUTS:
    nvec = (S,) vector, exogenous labor supply values (n_1,n_2,...n_S)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    L = scalar > 0, aggregate labor

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L
    --------------------------------------------------------------------
    '''
    L = nvec.sum()

    return L


def get_C(carr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    carr = (S,) vector or (S, T) matrix, distribution of consumption c_s
           in steady state or time path for the distribution of
           consumption

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    C = scalar > 0 or (T,) vector, aggregate consumption or time path of
        aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C


def get_cvec(r, w, zeta_s, BQ, bvec, nvec):
    '''
    --------------------------------------------------------------------
    Generates vector of lifetime steady-state consumptions given savings
    decisions, parameters, and the corresponding steady-state interest
    rate and wage.
    --------------------------------------------------------------------
    INPUTS:
    r    = scalar > 0 or (p,) vector, steady-state interest rate or p
           remaining periods of interest rates
    w    = scalar > 0 or (p,) vector, steady-state wage or p remaining
           periods of interest rates
    bvec = (p + 1,) vector, steady-state distribution of savings b_{s+1}
    nvec = (p,) vector, exogenous labor supply n_{s} in remaining
           periods
    zeta_s = (S, ) vector
    BQ   = scalar or (p, ) vector

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    b_s     = (p,) vector, b_init in first element and bvec in last p-1
              elements
    b_sp1   = (p,) vector, bvec in first p-1 elements and 0 in last
              element
    cvec    = (p,) vector, consumption by age c_s
    c_cnstr = (p,) Boolean vector, =True if c_s <= 0

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cvec, c_cnstr
    --------------------------------------------------------------------
    '''
    b_s = bvec[ : -1]
    b_sp1 = bvec[1:]
    cvec = (1 + r) * b_s + w * nvec + zeta_s[-len(b_s) : ] * BQ - b_sp1
    if cvec.min() <= 0:
        print('get_cvec() warning: distribution of savings and/or ' +
              'parameters created c<=0 for some agent(s)' +
              str(cvec[0]))
    c_cnstr = cvec <= 0

    return cvec, c_cnstr
    
    
def get_Y(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate output Y or time path of aggregate
    output Y_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 2 tuple, production function parameters
             (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0,1), capital share of income
    K      = scalar > 0 or (T+S-2,) vector, aggregate capital stock
             or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, aggregate labor or time
             path of the aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Y = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
        time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y


def feasible(params, bvec):
    '''
    --------------------------------------------------------------------
    Check whether a vector of steady-state savings is feasible in that
    it satisfies the nonnegativity constraints on consumption in every
    period c_s > 0 and that the aggregate capital stock is strictly
    positive K > 0
    --------------------------------------------------------------------
    INPUTS:
    params = length 4 tuple, (nvec, A, alpha, delta)
    nvec   = (S,) vector, exogenous labor supply values (n_1,n_2,...n_S)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    bvec   = (S,) vector, values of steady-state savings
             (b_2,b_3,...b_S)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_L()
        get_K()
        get_w()
        get_r()
        get_cvec()

    OBJECTS CREATED WITHIN FUNCTION:
    L        = scalar > 0, steady-state aggregate labor
    K        = scalar, steady-state aggregate capital stock
    K_cnstr  = Boolean, =True if K <= 0
    w_params = length 2 tuple, (A, alpha)
    w        = scalar, steady-state wage
    r_params = length 3 tuple, (A, alpha, delta)
    r        = scalar, steady-state interest rate
    bvec2    = (S,) vector, steady-state savings distribution plus
               initial period wealth of zero
    cvec     = (S,) vector, steady-state consumption by age
    c_cnstr  = (S,) Boolean vector, =True for elements for which c_s<=0
    b_cnstr  = (S-1,) Boolean, =True for elements for which b_s causes a
               violation of the nonnegative consumption constraint

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_cnstr, c_cnstr, K_cnstr
    --------------------------------------------------------------------
    '''
    nvec, A, alpha, delta, zeta_s = params
    L = get_L(nvec)
    K, K_cnstr = get_K(bvec)
    if not K_cnstr:
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        bvec2 = np.append([0], bvec)
        BQ = (1 + r) * bvec2[-1]
        cvec, c_cnstr = get_cvec(r, w, zeta_s, BQ, bvec2, nvec)
        b_cnstr = c_cnstr[:-1] + c_cnstr[1:]
        b_cnstr = np.append(b_cnstr, c_cnstr[-1])

    else:
        c_cnstr = np.ones(cvec.shape[0], dtype=bool)
        b_cnstr = np.ones(cvec.shape[0], dtype=bool)

    return b_cnstr, c_cnstr, K_cnstr
    
    
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
              
    
def get_b_errors(params, r, cvec, bvec, c_cnstr, diff):
    '''
    --------------------------------------------------------------------
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings decision. Because this function is used for
    solving for lifetime decisions in both the steady-state and in the
    transition path, lifetimes will be of varying length. Lifetimes in
    the steady-state will be S periods. Lifetimes in the transition path
    will be p in [2, S] periods
    --------------------------------------------------------------------
    INPUTS:
    params  = length 2 tuple, (beta, sigma)
    beta    = scalar in (0,1), discount factor
    sigma   = scalar > 0, coefficient of relative risk aversion
    r       = scalar > 0 or (p,) vector, steady-state interest rate or
              time path of interest rates
    cvec    = (p,) vector, distribution of consumption by age c_p
    bvec    = (p + 1,) vector
    c_cnstr = (p,) Boolean vector, =True if c<=0 for given bvec
    diff    = boolean, =True if use simple difference Euler
              errors. Use percent difference errors otherwise.

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mu_c     = (p-1,) vector, marginal utility of current consumption
    mu_cp1   = (p-1,) vector, marginal utility of next period consumpt'n
    mu_bq    = scalar, marginal benefit of bequest
    b_errors = (p,) vector, Euler errors characterizing optimal
               savings bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    beta, sigma, chib = params
    # Make each negative consumption artifically positive
    cvec[c_cnstr] = 9999.
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    mu_bq = chib * bvec[-1] ** (-sigma)
    if diff:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
        b_errors[c_cnstr[:-1]] = 9999.
        b_errors[c_cnstr[1:]] = 9999.
        b_errors = np.append(b_errors, mu_cp1[-1] - mu_bq)
        if c_cnstr[-1]:
            b_errors[-1] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_cnstr[:-1]] = 9999. / 100
        b_errors[c_cnstr[1:]] = 9999. / 100
        b_errors = np.append(b_errors, mu_cp1[-1] / mu_bq - 1)
        if c_cnstr[-1]:
            b_errors[-1] = 9999. / 100

    return b_errors
    
    
def EulerSys(bvec, *args):
    '''
    --------------------------------------------------------------------
    Generates vector of all Euler errors that characterize optimal
    lifetime decisions
    --------------------------------------------------------------------
    INPUTS:
    bvec    = (S,) vector, distribution of savings b_{s+1}
    args    = length 8 tuple,
              (beta, sigma, nvec, L, A, alpha, delta, EulDiff)
    beta    = scalar in [0,1), discount factor
    sigma   = scalar > 0, coefficient of relative risk aversion
    nvec    = (S,) vector, exogenous labor supply n_{s}
    L       = scalar > 0, exogenous aggregate labor
    A       = scalar > 0, total factor productivity parameter in firms'
              production function
    alpha   = scalar in (0,1), capital share of income
    delta   = scalar in [0,1], model-period depreciation rate of capital
    EulDiff = Boolean, =True if want difference version of Euler errors
              beta*(1+r)*u'(c2) - u'(c1), =False if want ratio version
              [beta*(1+r)*u'(c2)]/[u'(c1)] - 1

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_K()
        get_r()
        get_w()
        get_cvec()
        get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    K            = scalar > 0, aggregate capital stock
    K_constr     = Boolean, =True if K<=0 for given bvec
    b_err_vec    = (S-1,) vector, vector of Euler errors
    r_params     = length 3 tuple, (A, alpha, delta)
    r            = scalar > 0, interest rate
    w_params     = length 2 tuple, (A, alpha)
    w            = scalar > 0, wage
    bvec2        = (S,) vector, steady-state savings distribution plus
                   initial period wealth of zero
    cvec         = (S,) vector, consumption c_s for each age-s agent
    c_constr     = (S,) Boolean vector, =True if c<=0 for given bvec
    b_err_params = length 2 tuple, (beta, sigma)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    beta, sigma, chib, zeta_s, nvec, L, A, alpha, delta, EulDiff = args
    K, K_cnstr = get_K(bvec)
    if K_cnstr:
        b_err_vec = 1000. * np.ones(nvec.shape[0])
    else:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        bvec2 = np.append([0], bvec)
        BQ = (1 + r) * bvec2[-1]
        cvec, c_cnstr = get_cvec(r, w, zeta_s, BQ, bvec2, nvec)
        b_err_params = (beta, sigma, chib)
        b_err_vec = get_b_errors(b_err_params, r, cvec, bvec2, c_cnstr,
                                 EulDiff)

    return b_err_vec
    
def get_SS(params, bvec_guess, graphs):
    '''
    --------------------------------------------------------------------
    Solve for the steady-state solution of the S-period-lived agent OG
    model with exogenous labor supply
    --------------------------------------------------------------------
    INPUTS:
    params     = length 9 tuple, (beta, sigma, nvec, L, A, alpha, delta,
                 SS_tol, EulDiff)
    beta       = scalar in (0,1), discount factor for each model period
    sigma      = scalar > 0, coefficient of relative risk aversion
    nvec       = (S,) vector, exogenous labor supply n_{s,t}
    L          = scalar > 0, exogenous aggregate labor
    A          = scalar > 0, total factor productivity parameter in
                 firms' production function
    alpha      = scalar in (0,1), capital share of income
    delta      = scalar in [0,1], model-period depreciation rate of
                 capital
    SS_tol     = scalar > 0, tolerance level for steady-state fsolve
    EulDiff    = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                 version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    bvec_guess = (S-1,) vector, initial guesses for steady-state savings
                 (b_2, b_3, ...b_S)
    graphs     = Boolean, =True if output steady-state graphs

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    f_params     = length 4 tuple, (nvec, A, alpha, delta)
    b1_cnstr     = (S-1,) Boolean vector, =True if b_s causes negative
                   consumption c_s <= 0 or negative aggregate capital
                   stock K <= 0
    c1_cnstr     = (S,) Boolean vector, =True for elements of negative
                   consumption c_s <= 0
    K1_cnstr     = Boolean, =True if K <= 0
    eul_args     = length 8 tuple, (beta, sigma, nvec, L, A, alpha,
                   delta, EulDiff)
    b_ss         = (S-1,) vector, steady-state distribution of savings
    K_ss         = scalar > 0, steady-state aggregate capital stock
    K_cnstr      = Boolean, =True if K_ss <= 0
    r_params     = length 3 tuple, (A, alpha, delta)
    r_ss         = scalar > 0, steady-state interest rate
    w_params     = length 2 tuple, (A, alpha)
    w_ss         = scalar > 0, steady-state wage
    b_ss2        = (S,) vector, steady-state savings distribution plus
                   initial period wealth of zero
    c_ss         = (S,) vector, steady-state distribution of consumption
    c_cnstr      = (S,) Boolean vector,
    Y_params     = length 2 tuple, (A, alpha)
    Y_ss         = scalar > 0, steady-state aggregate output (GDP)
    C_ss         = scalar > 0, steady-state aggregate consumption
    b_err_params = length 2 tuple, (beta, sigma)
    EulErr_ss    = (S-1,) vector, steady-state Euler errors
    RCerr_ss     = scalar, resource constraint error Y_t - C_t - I_t
    ss_time      = scalar > 0, time elapsed in seconds
    ss_output    = length 10 dictionary, {b_ss, c_ss, w_ss, r_ss, K_ss,
                   Y_ss, C_ss, EulErr_ss, RCerr_ss, ss_time}

    FILES CREATED BY THIS FUNCTION:
        SS_bc.png

    RETURNS: ss_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    beta, sigma, chib, zeta_s, nvec, L, A, alpha, delta, SS_tol, EulDiff = params
    f_params = (nvec, A, alpha, delta, zeta_s)
    b1_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec_guess)
    if K1_cnstr is True or c1_cnstr.max() is True:
        err = ("Initial guess problem: " +
               "One or more constraints not satisfied.")
        print("K1_cnstr: ", K1_cnstr)
        print("c1_cnstr: ", c1_cnstr)
        raise RuntimeError(err)
    else:
        eul_args = (beta, sigma, chib, zeta_s, nvec, L, A, alpha, delta, EulDiff)
        b_ss = opt.fsolve(EulerSys, bvec_guess, args=(eul_args),
                          xtol=SS_tol)

    # Generate other steady-state values and Euler equations
    K_ss, K_cnstr = get_K(b_ss)
    r_params = (A, alpha, delta)
    r_ss = get_r(r_params, K_ss, L)
    w_params = (A, alpha)
    w_ss = get_w(w_params, K_ss, L)
    b_ss2 = np.append([0], b_ss)
    BQ_ss = (1 + r_ss) * b_ss[-1]
    c_ss, c_cnstr = get_cvec(r_ss, w_ss, zeta_s, BQ_ss, b_ss2, nvec)
    Y_params = (A, alpha)
    Y_ss = get_Y(Y_params, K_ss, L)
    C_ss = get_C(c_ss)
    b_err_params = (beta, sigma, chib)
    EulErr_ss = get_b_errors(
        b_err_params, r_ss, c_ss, b_ss, c_cnstr, EulDiff)
    RCerr_ss = Y_ss - C_ss - delta * K_ss

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
        age_pers = np.arange(1, S + 2)
        fig, ax = plt.subplots()
        plt.plot(age_pers, np.append(c_ss, 0), marker='D', label='Consumption')
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