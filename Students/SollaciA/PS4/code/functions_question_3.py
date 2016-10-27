import time
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

def get_SS(params, bvec_guess, nvec_guess, graphs):

    start_time = time.clock()
    S, beta, sigma, A, alpha, delta, chi_n_vec, b_ellip, mu, l_tilde, EulDiff, SS_tol = params
    f_params = (l_tilde, A, alpha, delta)
    bnvec_guess = np.append(bvec_guess, nvec_guess)
    n_low, n_high, b_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec_guess, nvec_guess)
    if K1_cnstr is True or c1_cnstr.max() is True:
        err = ("Initial guess problem: " +
               "One or more constraints not satisfied.")
        print("K1_cnstr: ", K1_cnstr)
        print("c1_cnstr: ", c1_cnstr)
        raise RuntimeError(err)
    else:
        eul_args = (S, beta, sigma, A, alpha, delta, chi_n_vec, b_ellip, mu, l_tilde, EulDiff)
        bn_ss = opt.fsolve(EulerSys, x0 = bnvec_guess, args=(eul_args),
                          xtol=SS_tol)
        b_ss = bn_ss[:S-1]
        n_ss = bn_ss[S-1:]

    K_ss, K_cnstr = get_K(b_ss)
    L_ss = get_L(n_ss)
    r_params = (A, alpha, delta)
    r_ss = get_r(r_params, K_ss, L_ss)
    w_params = (A, alpha)
    w_ss = get_w(w_params, K_ss, L_ss)
    b_ss2 = np.append([0], b_ss)
    c_ss, c_cnstr = get_cvec(r_ss, w_ss, b_ss2, n_ss)
    Y_params = (A, alpha)
    Y_ss = get_Y(Y_params, K_ss, L_ss)
    C_ss = get_C(c_ss)
    b_err_params = (beta, sigma)
    b_EulErr_ss = get_b_errors(
        b_err_params, r_ss, c_ss, c_cnstr, EulDiff)
    n_err_params = (sigma, chi_n_vec, b_ellip, mu, l_tilde)
    n_EulErr_ss = get_n_errors(n_err_params, w_ss, c_ss, c_cnstr, n_ss)
    EulErr_ss = np.append(b_EulErr_ss, n_EulErr_ss)
    RCerr_ss = Y_ss - C_ss - delta * K_ss

    ss_time = time.clock() - start_time

    ss_output = {
        'n_ss': n_ss,
        'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss,
        'K_ss': K_ss, 'L_ss': L_ss, 'Y_ss': Y_ss, 'C_ss': C_ss,
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
        S = n_ss.shape[0]
        age_pers = np.arange(1, S + 1)
        fig, ax = plt.subplots()
        plt.plot(age_pers, c_ss, marker='D', label='Consumption')
        plt.plot(age_pers, np.hstack((0, b_ss)), marker='D',
                 label='Savings')
#        plt.plot(age_pers, n_ss, marker='D',
#                 label='Labor')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Steady-state consumption and savings', fontsize=20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Units of consumption')
        plt.xlim((0, S + 1))
#        plt.ylim((-0.05, 1.2))
#        3.5 * (b_ss.max())
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "SS_bc")
        plt.savefig(output_path)
        plt.show()
        
        plt.plot(age_pers, n_ss, marker='D',
                 label='Labor')
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('steady-state labor supply', fontsize = 20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Labor Supply $n$')
        output_path = os.path.join(output_dir, "SS_n")
        plt.savefig(output_path)

    return ss_output
    


def feasible(params, bvec_guess, nvec_guess):
    l_tilde, A, alpha, delta = params
    L = get_L(nvec_guess)
    K, K_cnstr = get_K(bvec_guess)
    if not K_cnstr:
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        bvec2 = np.append([0], bvec_guess)
        cvec, c_cnstr = get_cvec(r, w, bvec2, nvec_guess)
        b_cnstr = c_cnstr[:-1] + c_cnstr[1:]
        n_low = nvec_guess <= 0
        n_high = nvec_guess > l_tilde

    else:
        c_cnstr = np.ones(cvec.shape[0], dtype=bool)
        b_cnstr = np.ones(cvec.shape[0] - 1, dtype=bool)
        n_low = np.ones(cvec.shape[0], dtype=bool)
        n_high = np.ones(cvec.shape[0], dtype=bool)

    return n_low, n_high, b_cnstr, c_cnstr, K_cnstr

def EulerSys(bn_vec, *args):

    S, beta, sigma, A, alpha, delta, chi_n_vec, b_ellip, mu, l_tilde, EulDiff = args
    bvec = bn_vec[:S-1]
    nvec = bn_vec[S-1:]
    K, K_cnstr = get_K(bvec)
    L = get_L(nvec)
    if K_cnstr:
        err_vec = 1000. * np.ones(2*S - 1)
    else:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        bvec2 = np.append([0], bvec)
        cvec, c_cnstr = get_cvec(r, w, bvec2, nvec)
        b_err_params = (beta, sigma)
        b_err_vec = get_b_errors(b_err_params, r, cvec, c_cnstr,
                                 EulDiff)
        n_err_params = (sigma, chi_n_vec, b_ellip, mu, l_tilde)
        n_err_vec = get_n_errors(n_err_params, w, cvec, c_cnstr, nvec)
        err_vec = np.append(b_err_vec, n_err_vec)
    return err_vec    
    
    
def get_b_errors(params, r, cvec, c_cnstr, diff):

    beta, sigma = params
    # Make each negative consumption artifically positive
    cvec[c_cnstr] = 9999.
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    if diff:
        b_errors = (beta * (1 + r) * mu_cp1) - mu_c
        b_errors[c_cnstr[:-1]] = 9999.
        b_errors[c_cnstr[1:]] = 9999.
    else:
        b_errors = ((beta * (1 + r) * mu_cp1) / mu_c) - 1
        b_errors[c_cnstr[:-1]] = 9999. / 100
        b_errors[c_cnstr[1:]] = 9999. / 100

    return b_errors

 
def get_n_errors(params, w, cvec, c_cnstr, nvec):

    sigma, chi_n_vec, b_ellip, mu, l_tilde = params
    # Make each negative consumption artifically positive
    cvec[c_cnstr] = 9999.
    n_low = nvec <= 0
    n_high = nvec >= 1
    mu_c = cvec ** (-sigma)
    mu_n = chi_n_vec * (b_ellip / l_tilde) * (nvec / l_tilde) ** (mu - 1)*(1 - (nvec / l_tilde) ** mu) ** ((1 - mu) / mu)
    n_errors = mu_c*w - mu_n
    n_errors[n_low] = 999
    n_errors[n_high] = 999
    
    return n_errors
    
def get_r(params, K, L):

    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r
    
def get_w(params, K, L):

    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w
    

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
            print('Aggregate capital constraint is violated K<=0 for ' +
                  'some period in time path.')

    return K, K_cnstr
    
    
def get_L(arr):

    if arr.ndim == 1:
        L = arr.sum()
    elif arr.ndim == 2:
        L = arr.sum(axis = 0)

    return L


def get_C(carr):

    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C


def get_cvec(r, w, bvec, nvec):

    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    cvec = (1 + r) * b_s + w * nvec - b_sp1
    c_cnstr = cvec <= 0

    return cvec, c_cnstr
    
    
def get_Y(params, K, L):

    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y
    
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