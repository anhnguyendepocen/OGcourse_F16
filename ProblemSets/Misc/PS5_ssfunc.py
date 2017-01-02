# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import os

def get_L(nvec):
    
    L = nvec.sum()
    return L

def get_K(barr):
    
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
            print('Aggregate capital cnstraint is violated K<=0 for ' +
                  'some period in time path.')
    return K, K_cnstr

def get_Y(params, K, L):
    
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))
    return Y

def get_C(carr):
   
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)
    return C


def get_r(params, K, L):
   
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta
    return r


def get_w(params, K, L):
   
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)
    return w


def get_BQ(r, bvec):

    bvec2 = np.append([0], bvec)
    BQ = (1 + r) * bvec2[-1]
    return BQ


def get_cvec_ss(r, w,BQ, zeta_s, bvec, nvec):
    
    b_s = bvec[ : -1]
    b_sp1 = bvec[1:]
    cvec = (1 + r) * b_s + w * nvec + zeta_s * BQ - b_sp1
    if cvec.min() <= 0:
        print('get_cvec() warning: distribution of savings and/or ' +
              'parameters created c<=0 for some agent(s)' +
              str(cvec[0]))
    c_cnstr = cvec <= 0

    return cvec, c_cnstr

def feasible(params, bvec):

    nvec, A, alpha, delta, zeta_s = params
    L = get_L(nvec)
    K, K_cnstr = get_K(bvec)
    if not K_cnstr:
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        bvec2 = np.append([0], bvec)
        # BQ = (1 + r) * bvec2[-1]
        BQ = get_BQ(r, bvec)
        cvec, c_cnstr = get_cvec_ss(r, w, BQ, zeta_s, bvec2, nvec)
        b_cnstr = c_cnstr[:-1] + c_cnstr[1:]
        b_cnstr = np.append(b_cnstr, c_cnstr[-1])

    else:
        c_cnstr = np.ones(cvec.shape[0], dtype=bool)
        b_cnstr = np.ones(cvec.shape[0], dtype=bool)

    return b_cnstr, c_cnstr, K_cnstr
    
    
def print_time(seconds, type):

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
              
    

def get_b_errors(params, r, bvec, cvec, c_cnstr):

    beta, sigma, chi_b = params
    # Make each negative consumption artifically positive
    cvec[c_cnstr] = 9999.
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    mu_bq = chi_b * bvec[-1] ** (-sigma)

    b_errors = (beta * (1 + r) * mu_cp1) - mu_c
    b_errors[c_cnstr[:-1]] = 9999.
    b_errors[c_cnstr[1:]] = 9999.
    b_errors = np.append(b_errors, mu_cp1[-1] - mu_bq)
    if c_cnstr[-1]:
        b_errors[-1] = 9999.

    return b_errors


def EulerSys(bvec, *args):   
    beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta = args
    K, K_cnstr = get_K(bvec)
    if K_cnstr == True:
        b_err_vec = 1000 * np.ones(80)
        # b_err_vec = 1000. * np.ones(nvec.shape[0])
    else:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        bvec2 = np.append([0], bvec)
        BQ = get_BQ(r, bvec2[-1])
        cvec, c_cnstr = get_cvec_ss(r, w,BQ, zeta_s, bvec2, nvec)
        b_err_params = (beta, sigma, chi_b)
        b_err_vec = get_b_errors(b_err_params, r, bvec2, cvec, c_cnstr)
    return b_err_vec



def SS(params, bvec_guess, graphs):

    start_time = time.clock()
    beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta, SS_tol = params
    
    eul_args = beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta
    b_ss = opt.fsolve(EulerSys, bvec_guess, args=(eul_args), xtol=SS_tol)

    # Generate other steady-state values and Euler equations
    K_ss, K_cnstr = get_K(b_ss)
    r_params = (A, alpha, delta)
    r_ss = get_r(r_params, K_ss, L)
    w_params = (A, alpha)
    w_ss = get_w(w_params, K_ss, L)
    b_ss2 = np.append([0], b_ss)
    BQ_ss = get_BQ(r_ss, b_ss2[-1])
    c_ss, c_cnstr = get_cvec_ss(r_ss, w_ss, BQ_ss, zeta_s, b_ss2, nvec)
    Y_params = (A, alpha)
    Y_ss = get_Y(Y_params, K_ss, L)
    C_ss = get_C(c_ss)
    b_err_params = (beta, sigma, chi_b)
    EulErr_ss = get_b_errors(b_err_params, r_ss, b_ss, c_ss, c_cnstr)
    RCerr_ss = Y_ss - C_ss - delta * K_ss


    ss_time = time.clock() - start_time

    ss_output = {
        'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
        'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss,
        'ss_time': ss_time}
    print('b_ss is: ', b_ss)
    print('c_ss is:', c_ss)
    print('w_ss is:', w_ss)
    print('r_ss is:', r_ss)
    print('K_ss is:', K_ss)
    print('Y_ss is:', Y_ss)    
    print('C_ss is:', C_ss)
    print('Euler errors are: ', EulErr_ss)
    print('Resource constraint error is: ', RCerr_ss)


    # Print SS computation time
    print_time(ss_time, 'SS')

    if graphs:

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
        plt.show()

    return ss_output

