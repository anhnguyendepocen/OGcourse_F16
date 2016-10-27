# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time
import sys


#1 
def MU_n_sse(ELP_params, *args):
    '''
    --------------------------------------------------------------------
    Calculates the sum of squared differences between a vector of CRRA
    marginal disutilities of labor supply and a vector of CFE marginal
    disutilities of labor supply
    --------------------------------------------------------------------
    INPUTS:
    CFE_params = (2,) vector, values for chi and theta from CFE
                 disutility of labor supply function
    chi        = scalar > 0, level parameter for CFE function
    theta      = scalar > 0, shape parameter (Frisch elasticity of labor
                 supply) for CFE function
    args       = length 3 tuple, (nvec, CRRA_scale, sigma)
    nvec       = (N,) vector, values in support of labor supply
    CRRA_scale = scalar > 0, scale parameter in CRRA function
    sigma      = scalar > 1, shape parameter (coefficient of constant
                 relative risk aversion) in CRRA function

    OTHER FUNCTIONS CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN THIS FUNCTION:
    MU_CRRA   = (N,) vector, CRRA marginal utilities of labor supply
    MU_CFE    = (N,) vector, CFE marginal utilities of labor supply
    criterion = scalar > 0, sum of squared differences between MU_CFE
                and MU_CRRA

    RETURNS: criterion
    --------------------------------------------------------------------
    '''
    nvec, chi, theta_init = args
    b_init, mu_init = ELP_params
    MU_ELP = (b_init / mu_init) * ((1- nvec) ** mu_init) ** ((1 - mu_init)/ mu_init)
    MU_CFE = chi * (nvec) ** (1 / theta_init)
    criterion = ((MU_CFE - MU_ELP) ** 2).sum()


    return criterion


def feasible(f_params, nvec_guess, bvec_guess):
    l_tilde, A, alpha, delta = f_params
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

    return b_cnstr, c_cnstr, K_cnstr, n_low, n_high


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
            print('Aggregate capital constraint is violated K<=0 for ' +
                  'some period in time path.')

    return K, K_cnstr


def get_w(w_params, K, L):
    A, alpha = w_params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w


def get_r(params, K, L):
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r
   

def get_cvec(r, w, bvec, nvec):
    bt = bvec
    bt1 = np.append(bvec[1:], [0])
    cvec = (1 + r) * bt + w * nvec - bt1

    c_cnstr = cvec <= 0

    return cvec, c_cnstr

def get_C(carr):
    
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C

def get_Y(params, K, L):
    
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y

def get_b_errors(params, r, cvec, c_cnstr):
    
    beta, sigma = params
    # Make each negative consuTmption artifically positive
    cvec[c_cnstr] = 9999.
    mu_c = cvec[:-1] ** (-sigma)
    mu_cp1 = cvec[1:] ** (-sigma)
    b_errors = (beta * (1 + r) * mu_cp1) - mu_c
    b_errors[c_cnstr[:-1]] = 9999.
    b_errors[c_cnstr[1:]] = 9999.

    return b_errors


def n_mu(params, w, nvec):
    chi, b, mu, l_tilde = params
    mu_n = 1 * chi * (b / l_tilde) * (nvec / l_tilde) ** (mu - 1) * \
        (1 - (nvec / l_tilde) ** mu) ** ((1 - mu) / mu)
    return mu_n

def get_n_errors(params, w, cvec, c_cnstr, nvec):
    sigma, chi, b, mu, l_tilde = params
    n_param = (chi, b, mu, l_tilde)
    n_low = nvec <= 0
    n_high = nvec >= 1
    cvec[c_cnstr] = 999.
    mu_c = cvec ** (-sigma)
    mu_n = n_mu(n_param, w, nvec)
    n_errors = w * mu_c - mu_n
    n_errors[n_low] = 999
    n_errors[n_high] = 999
    return n_errors


def EulerSys(bnvec, *args):
    S, beta, sigma, A, alpha, delta, chi, b, mu, l_tilde = args
    bvec = bnvec[: S - 1]
    nvec = bnvec[S - 1 :]
    K, K_cnstr = get_K(bvec)
    L = get_L(nvec)
    if K_cnstr:
        b_err_vec = 1000. * np.ones(nvec.shape[0] - 1)
        n_err_vec = 1000. * np.ones(nvec.shape[0] )
    else:
        r_params = (A, alpha, delta)
        r = get_r(r_params, K, L)
        w_params = (A, alpha)
        w = get_w(w_params, K, L)
        bvec2 = np.append([0], bvec)
        cvec, c_cnstr = get_cvec(r, w, bvec2, nvec)
        b_err_params = (beta, sigma)
        b_err_vec = get_b_errors(b_err_params, r, cvec, c_cnstr)
        n_err_params = (sigma, chi, b, mu, l_tilde)
        n_err_vec = get_n_errors(n_err_params, w, cvec, c_cnstr, nvec)
    error = np.append(b_err_vec, n_err_vec)
    #    print(error)
    return error

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


def get_SS(params, bvec, nvec, graphs):
    start_time = time.clock()
    S, beta, sigma, A, alpha, delta, chi, b, mu, l_tilde, SS_tol = params
    f_params = (l_tilde, A, alpha, delta)
#    n_low, n_high, b_cnstr, c1_cnstr, K1_cnstr = feasible(f_params, bvec, nvec)
    guess = np.append(bvec, nvec)
    print(guess)
    # if K1_cnstr is True or c1_cnstr.max() is True:
    #     err = ("Initial guess problem: " +
    #            "One or more constraints not satisfied.")
    #     print("K1_cnstr: ", K1_cnstr)
    #     print("c1_cnstr: ", c1_cnstr)
    #     raise RuntimeError(err)
    # else:
    L = get_L(nvec)
    eul_args = (S, beta, sigma, A, alpha, delta, chi, b, mu, l_tilde)
    ss = opt.fsolve(EulerSys, guess, args=(eul_args), xtol=SS_tol)
    b_ss = ss[: S - 1]
    print(len(b_ss))
    n_ss = ss[S - 1:]
    print(len(n_ss))

    # Generate other steady-state values and Euler equations
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
        b_err_params, r_ss, c_ss, c_cnstr)
    n_err_params = (sigma, chi, b, mu, l_tilde)
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
#     
        plt.legend(loc='upper left')
        output_path = os.path.join(output_dir, "SS_bc")
        plt.savefig(output_path)
        plt.show()
        
        output_path = os.path.join(output_dir, "Labor_supply_ss")
        plt.plot(age_pers, n_ss, marker='D',
                 label='Labor')
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('steady-state labor supply', fontsize = 20)
        plt.xlabel(r'Age $s$')
        plt.ylabel(r'Labor Supply $n$')
        plt.savefig(output_path)

    return ss_output
    

# TPI
def get_path(x1, xT, T, spec):
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) + cc

    return xpath


def LfEulerSys(bn, *args):
   
    beta, sigma, chi, b, mu, l_tilde, beg_wealth, p, rpath, wpath= args
    bvec = bn[ : p - 1]
    nvec = bn[p - 1 : ]
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_cnstr = get_cvec(rpath, wpath, bvec2, nvec)
    b_err_params = (beta, sigma)
    b_err_vec = get_b_errors(b_err_params, rpath[1:], cvec,
                                   c_cnstr)
    n_err_params = (sigma, chi, b, mu, l_tilde)
    n_err_vec = get_n_errors(n_err_params, wpath, cvec, c_cnstr, nvec)
    err_vec = np.append(b_err_vec, n_err_vec)
    return err_vec

def paths_life(params, beg_age, beg_wealth, rpath, wpath, b_init, n_init):
    
    S, beta, sigma, chi, b, mu, l_tilde, TPI_tol= params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
#    if len(nvec) != p:
#        sys.exit("Beginning age and length of nvec do not match.")
    bguess =  b_init
    nguess = n_init
    guess = np.append(bguess, nguess)
    eullf_objs = (beta, sigma, chi, b, mu, l_tilde, beg_wealth, p, rpath, wpath)
    bnpath = opt.fsolve(LfEulerSys, guess, args=(eullf_objs), xtol=SS_tol)
    bpath = bnpath[ : p - 1]
    npath = bnpath[p - 1: ]
    cpath, c_cnstr = get_cvec(rpath, wpath,
                                    np.append(beg_wealth, bpath), npath)
    b_err_params = (beta, sigma)
    b_err_vec = get_b_errors(b_err_params, rpath[1:], cpath,
                                   c_cnstr)
    n_err_params = (sigma, chi, b, mu, l_tilde)
    n_err_vec = get_n_errors(n_err_params, wpath, cpath, c_cnstr, npath)
    return bpath, npath, cpath, b_err_vec, n_err_vec

def n_err_last(nvec, *args):
        b_last = bpath[-1, 0]
        params, b_last, w, r = args
        sigma, chi, b, mu, l_tilde = params
        n_err_params = (chi, b, mu, l_tilde)
        mu_n = mu_n(n_err_params, w, nvec)
        mu_c = w * (w * nvec + (1 + r) * b_last) ** (-sigma)
        return mu_n - mu_c

def get_cbnepath(params, rpath, wpath):
    S, T, beta, sigma,  chi, b, mu, l_tilde, bvec1, b_ss, n_ss, SS_tol= params
    cpath = np.zeros((S, T + S - 2))
    bpath = np.append(bvec1.reshape((S - 1, 1)),
                      np.zeros((S - 1, T + S - 3)), axis=1)
    npath = np.zeros((S, T + S - 2))
    EulErrPath_inter = np.zeros((S - 1, T + S - 2))
    EulErrPath_intra = np.zeros((S, T + S - 2))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
#    cpath[S - 1, 0] = ((1 + rpath[0]) * bvec1[S - 2] +\
#                       wpath[0] * nvec[S - 1])
    # get the labor saving decision for the oldest agent
    b_last = bpath [-1, 0]
    w0 = wpath[0]
    r0 = rpath[0]
    n_err_last_params = (sigma, chi, b_par, miu, l_endow)
    n_last = opt.fsolve(n_err_last, x0 = n_ss[-1], args = (n_err_last_params, b_last, w0, r0), xtol = TPI_tol)

    cpath[S - 1,  0] = w0 * n_last + (1 + r0) * b_last
    npath[-1, 0] = n_last
    print(cpath[-1, 0], w0, n_last, r0, b_last)
    pl_params = (S, beta, sigma, chi, b, mu, l_tilde, SS_tol)
    for p in range(2, S):
        b_guess = np.diagonal(bpath[S - p:, :p - 1])
        n_guess = np.append([n_ss[S - p]], np.diagonal(npath[S - p + 1:, : p - 1]))
#        n_guess = 0.5 * n_ss[S - p: ] + 0.5
#        n_guess = n_ss[S - p:]
#        print(n_guess)
        bveclf, nveclf, cveclf, b_err_veclf, n_err_veclf = paths_life(
            pl_params, S - p + 1, bvec1[S - p - 1], 
            rpath[:p], wpath[:p], b_guess, n_guess)
#        print(len(wpath[:p]), len(nveclf), len(bveclf))
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        DiagMaskb = np.eye(p - 1, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        bpath[S - p:, 1:p] = DiagMaskb * bveclf + bpath[S - p:, 1:p]
        cpath[S - p:, :p] = DiagMaskc * cveclf + cpath[S - p:, :p]
#        print(cpath[-1, 0])
        npath[S - p:, :p] = DiagMaskc * nveclf + npath[S - p:, :p]
        EulErrPath_inter[S - p:, 1:p] = (DiagMaskb * b_err_veclf +
                                   EulErrPath_inter[S - p:, 1:p])
        EulErrPath_intra[S - p:, :p] = (DiagMaskc * n_err_veclf +
                                    EulErrPath_intra[S - p, :p])
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    DiagMaskb = np.eye(S - 1, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    for t in range(1, T):  # Go from periods 1 to T-1
        b_guess = np.diagonal(bpath[:, t - 1:t + S - 2])
        if t == 1:
            n_guess = np.append(n_ss[0], np.diagonal(npath[1:, t - 1 :t + S - 2]))
        else:
            n_guess = np.diagonal(npath[:, t - 2 :t + S - 2])
        bveclf, nveclf, cveclf, b_err_veclf, n_err_veclf = paths_life(
            pl_params, 1, 0, rpath[t - 1:t + S - 1],
            wpath[t - 1:t + S - 1], b_guess, n_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        bpath[:, t:t + S - 1] = (DiagMaskb * bveclf +
                                 bpath[:, t:t + S - 1])
        cpath[:, t - 1:t + S - 1] = (DiagMaskc * cveclf +
                                     cpath[:, t - 1:t + S - 1])
        npath[:, t - 1: t+ S - 1] = (DiagMaskc * nveclf + 
                                        npath[:, t - 1 : t+ S - 1])
        EulErrPath_inter[:, t:t + S - 1] = (DiagMaskb * b_err_veclf +
                                      EulErrPath_inter[:, t:t + S - 1])
        EulErrPath_intra[:, t - 1: t + S - 1] = (DiagMaskc * n_err_veclf +
                                        EulErrPath_intra[:, t - 1 : t + S -1])

    return cpath, bpath, npath, EulErrPath_inter, EulErrPath_intra



# def get_TPI(params, bvec1, graphs):
#     start_time = time.clock()
#     (S, T, beta, sigma, chi, b, mu, l_tilde, A, alpha, delta, b_ss, n_ss, K_ss, C_ss,
#         L_ss, maxiter_TPI, mindist_TPI, xi, SS_tol) = params
#     K1, K1_cnstr = get_K(bvec1)

#     # Create time paths for K and L
#     Kpath_init = np.zeros(T + S - 2)
#     Kpath_init[:T] = get_path(K1, K_ss, T, "linear")
#     Kpath_init[T:] = K_ss
#     Lpath_init = L_ss * np.ones(T + S - 2)

#     iter_TPI = int(0)
#     dist_TPI = 10.
#     Kpath_new = Kpath_init.copy()
#     Lpath_new = Lpath_init.copy()
#     r_params = (A, alpha, delta)
#     w_params = (A, alpha)
#     cbne_params = (S, T, beta, sigma, chi, b, mu, l_tilde, bvec1, b_ss, 
#                    n_ss, SS_tol)

#     while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
#         iter_TPI += 1
#         Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
#         Lpath_init = xi * Lpath_new + (1 - xi) * Lpath_init
#         rpath = get_r(r_params, Kpath_init, Lpath_init)
#         wpath = get_w(w_params, Kpath_init, Lpath_init)
#         cpath, bpath, npath, EulErrPath_inter, EulErrPath_intra = get_cbnepath(cbne_params, rpath, wpath)
#         Kpath_new = np.zeros(T + S - 2)
#         Kpath_new[:T], Kpath_cnstr = get_K(bpath[:, :T])
#         Kpath_new[T:] = K_ss * np.ones(S - 2)
#         Lpath_new[:T - 1] = get_L(npath[:, :T - 1])
#         Lpath_new[T - 1:] = L_ss * np.ones(S - 1)
#         Kpath_cnstr = np.append(Kpath_cnstr,
#                                 np.zeros(S - 2, dtype=bool))
#         Kpath_new[Kpath_cnstr] = 0.1

#         # Check the distance of Kpath_new1
#         dist_TPI = ((Kpath_new[1:T] - Kpath_init[1:T]) ** 2).sum() + ((Lpath_new[:T] - Lpath_init[:T]) ** 2).sum()
#         # dist_TPI = np.absolute((Kpath_new[1:T] - Kpath_init[1:T]) /
#         #                        Kpath_init[1:T]).max()
#         print('iter: ', iter_TPI, ', dist: ', dist_TPI,
#               ',max Eul err Intra: ', np.absolute(EulErrPath_intra).max(),
#               ',max Eul err Inter: ', np.absolute(EulErrPath_inter).max())

#     if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
#         print('TPI reached maxiter and did not converge.')
#     elif iter_TPI == maxiter_TPI and dist_TPI <= mindist_TPI:
#         print('TPI converged in the last iteration. ' +
#               'Should probably increase maxiter_TPI.')
#     Kpath = Kpath_new
#     Lpath = Lpath_new
#     Y_params = (A, alpha)
#     Ypath = get_Y(Y_params, Kpath, Lpath)
#     Cpath = np.zeros(T + S - 2)
#     Cpath[:T - 1] = get_C(cpath[:, :T - 1])
#     Cpath[T - 1:] = C_ss * np.ones(S - 1)
#     RCerrPath = (Ypath[:-1] - Cpath[:-1] - Kpath[1:] +
#                  (1 - delta) * Kpath[:-1])
#     tpi_time = time.clock() - start_time

#     tpi_output = {
#         'bpath': bpath, 'cpath': cpath, 'npath': npath, 'wpath': wpath, 'rpath': rpath,
#         'Kpath': Kpath, 'Lpath': Lpath, 'Ypath': Ypath, 'Cpath': Cpath,
#         'EulErrPath IntraTemporal': EulErrPath_intra,'EulErrPath InterTemporal': EulErrPath_inter,
#         'RCerrPath': RCerrPath,
#         'tpi_time': tpi_time}

#     # Print TPI computation time
#     print_time(tpi_time, 'TPI')

#     if graphs:
#         '''
#         ----------------------------------------------------------------
#         cur_path    = string, path name of current directory
#         output_fldr = string, folder in current path to save files
#         output_dir  = string, total path of images folder
#         output_path = string, path of file name of figure to be saved
#         tvec        = (T+S-2,) vector, time period vector
#         tgridTm1    = (T-1,) vector, time period vector to T-1
#         tgridT      = (T,) vector, time period vector to T-1
#         sgrid       = (S,) vector, all ages from 1 to S
#         sgrid2      = (S-1,) vector, all ages from 2 to S
#         tmatb       = (S-1, T) matrix, time periods for all savings
#                       decisions ages (S-1) and time periods (T)
#         smatb       = (S-1, T) matrix, ages for all savings decision
#                       ages (S-1) and time periods (T)
#         tmatc       = (3, T-1) matrix, time periods for all consumption
#                       decisions ages (S) and time periods (T-1)
#         smatc       = (3, T-1) matrix, ages for all consumption
#                       decisions ages (S) and time periods (T-1)
#         ----------------------------------------------------------------
#         '''
#         # Create directory if images directory does not already exist
#         cur_path = os.path.split(os.path.abspath(__file__))[0]
#         output_fldr = "images"
#         output_dir = os.path.join(cur_path, output_fldr)
#         if not os.access(output_dir, os.F_OK):
#             os.makedirs(output_dir)

#         # Plot time path of aggregate capital stock
#         tvec = np.linspace(1, T + S - 2 , T + S - 2)
#         minorLocator = MultipleLocator(1)
#         fig, ax = plt.subplots()
#         plt.plot(tvec, Kpath, marker='D')
#         # for the minor ticks, use no labels; default NullFormatter
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         plt.title('Time path for aggregate capital stock K')
#         plt.xlabel(r'Period $t$')
#         plt.ylabel(r'Aggregate capital $K_{t}$')
#         output_path = os.path.join(output_dir, "Kpath")
#         plt.savefig(output_path)
#         plt.show()
        
#         plt.plot(tvec, Lpath, marker='D')
#         # for the minor ticks, use no labels; default NullFormatter
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         plt.title('Time path for aggregate labor supply L')
#         plt.xlabel(r'Period $t$')
#         plt.ylabel(r'Aggregate labor supply $L_{t}$')
#         output_path = os.path.join(output_dir, "Lpath")
#         plt.savefig(output_path)
#         plt.show()

#         # Plot time path of aggregate output (GDP)
#         fig, ax = plt.subplots()
#         plt.plot(tvec, Ypath, marker='D')
#         # for the minor ticks, use no labels; default NullFormatter
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         plt.title('Time path for aggregate output (GDP) Y')
#         plt.xlabel(r'Period $t$')
#         plt.ylabel(r'Aggregate output $Y_{t}$')
#         output_path = os.path.join(output_dir, "Ypath")
#         plt.savefig(output_path)
#         # plt.show()

#         # Plot time path of aggregate consumption
#         fig, ax = plt.subplots()
#         plt.plot(tvec, Cpath, marker='D')
#         # for the minor ticks, use no labels; default NullFormatter
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         plt.title('Time path for aggregate consumption C')
#         plt.xlabel(r'Period $t$')
#         plt.ylabel(r'Aggregate consumption $C_{t}$')
#         output_path = os.path.join(output_dir, "C_aggr_path")
#         plt.savefig(output_path)
#         # plt.show()

#         # Plot time path of real wage
#         fig, ax = plt.subplots()
#         plt.plot(tvec, wpath, marker='D')
#         # for the minor ticks, use no labels; default NullFormatter
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         plt.title('Time path for real wage w')
#         plt.xlabel(r'Period $t$')
#         plt.ylabel(r'Real wage $w_{t}$')
#         output_path = os.path.join(output_dir, "wpath")
#         plt.savefig(output_path)
#         # plt.show()

#         # Plot time path of real interest rate
#         fig, ax = plt.subplots()
#         plt.plot(tvec, rpath, marker='D')
#         # for the minor ticks, use no labels; default NullFormatter
#         ax.xaxis.set_minor_locator(minorLocator)
#         plt.grid(b=True, which='major', color='0.65', linestyle='-')
#         plt.title('Time path for real interest rate r')
#         plt.xlabel(r'Period $t$')
#         plt.ylabel(r'Real interest rate $r_{t}$')
#         output_path = os.path.join(output_dir, "rpath")
#         plt.savefig(output_path)
        # plt.show()

#         # Plot time path of individual savings distribution
#         tgridT = np.linspace(1, T, T)
#         sgrid2 = np.linspace(2, S, S - 1)
#         tmatb, smatb = np.meshgrid(tgridT, sgrid2)
#         cmap_bp = matplotlib.cm.get_cmap('summer')
#         fig = plt.figure()
#         ax = Axes3D(fig)
# #        ax = fig.gca(projection='3d')
#         ax.set_xlabel(r'period-$t$')
#         ax.set_ylabel(r'age-$s$')
#         ax.set_zlabel(r'individual savings $b_{s,t}$')
#         strideval = max(int(1), int(round(S / 10)))
#         ax.plot_surface(tmatb, smatb, bpath[:, :T], rstride=strideval,
#                         cstride=strideval, cmap=cmap_bp)
#         output_path = os.path.join(output_dir, "bpath")
#         plt.savefig(output_path)
#         # plt.show()

#         # Plot time path of individual consumption distribution
#         tgridTm1 = np.linspace(1, T - 1, T - 1)
#         sgrid = np.linspace(1, S, S)
#         tmatc, smatc = np.meshgrid(tgridTm1, sgrid)
#         cmap_cp = matplotlib.cm.get_cmap('summer')
#         fig = plt.figure()
#         ax = Axes3D(fig)
# #        ax = fig.gca(projection='3d')
#         ax.set_xlabel(r'period-$t$')
#         ax.set_ylabel(r'age-$s$')
#         ax.set_zlabel(r'individual consumption $c_{s,t}$')
#         strideval = max(int(1), int(round(S / 10)))
#         ax.plot_surface(tmatc, smatc, cpath[:, :T - 1],
#                         rstride=strideval, cstride=strideval,
#                         cmap=cmap_cp)
#         output_path = os.path.join(output_dir, "cpath")
#         plt.savefig(output_path)
#         # plt.show()

    return tpi_output

