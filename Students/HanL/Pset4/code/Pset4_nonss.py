# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:31:03 2016

@author: Luke
"""

# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import Chap7_ss as c7ssf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import sys
import os


def get_TPI(params, bvec1, graphs):
    '''
    --------------------------------------------------------------------
    Generates steady-state time path for all endogenous objects from
    initial state (K1, Gamma1) to the steady state.
    --------------------------------------------------------------------
    INPUTS:
    params      = length 17 tuple, (S, T, beta, sigma, nvec, L, A,
                  alpha, delta, b_ss, K_ss, C_ss, maxiter_TPI,
                  mindist_TPI, xi, TPI_tol, EulDiff)
    S           = integer in [3,80], number of periods an individual
                  lives
    T           = integer > S, number of time periods until steady
                  state
    beta        = scalar in (0,1), discount factor for model period
    sigma       = scalar > 0, coefficient of relative risk aversion
    nvec        = (S,) vector, exogenous labor supply n_{s,t}
    L           = scalar > 0, exogenous aggregate labor
    A           = scalar > 0, total factor productivity parameter in
                  firms' production function
    alpha       = scalar in (0,1), capital share of income
    delta       = scalar in [0,1], model-period depreciation rate of
                  capital
    b_ss        = (S-1,) vector, steady-state distribution of savings
    K_ss        = scalar > 0, steady-state aggregate capital stock
    C_ss        = scalar > 0, steady-state aggregate consumption
    maxiter_TPI = integer >= 1, Maximum number of iterations for TPI
    mindist_TPI = scalar > 0, Convergence criterion for TPI
    xi          = scalar in (0,1], TPI path updating parameter
    TPI_tol     = scalar > 0, tolerance level for fsolve's in TPI
    EulDiff     = Boolean, =True if want difference version of Euler
                  errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                  ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    bvec1       = (S-1,) vector, initial period savings distribution
    graphs      = Boolean, =True if want graphs of TPI objects

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        c5ssf.get_K()
        get_path()
        c5ssf.get_r()
        c5ssf.get_w()
        get_cbepath()
        c5ssf.get_Y()
        c5ssf.get_C()
        c5ssf.print_time()


    OBJECTS CREATED WITHIN FUNCTION:
    start_time  = scalar, current processor time in seconds (float)
    K1          = scalar > 0, initial aggregate capital stock
    K1_cnstr    = Boolean, =True if K1 <= 0
    Kpath_init  = (T+S-2,) vector, initial guess for the time path
                  of the aggregate capital stock
    Lpath       = (T+S-2,) vector, exogenous time path for aggregate
                  labor
    iter_TPI    = integer >= 0, current iteration of TPI
    dist_TPI    = scalar >= 0, distance measure for fixed point
    Kpath_new   = (T+S-2,) vector, new path of the aggregate capital
                  stock implied by household and firm optimization
    r_params    = length 3 tuple, (A, alpha, delta)
    w_params    = length 2 tuple, (A, alpha)
    cbe_params  = length 9 tuple, (S, T, beta, sigma, nvec, bvec1, b_ss,
                  TPI_tol, EulDiff)
    rpath       = (T+S-2,) vector, time path of the interest rate
    wpath       = (T+S-2,) vector, time path of the wage
    cpath       = (S, T+S-2) matrix, time path values of distribution of
                  consumption c_{s,t}
    bpath       = (S-1, T+S-2) matrix, time path of distribution of
                  individual savings b_{s,t}
    EulErrPath  = (S-1, T+S-2) matrix, time path of individual Euler
                  errors corresponding to individual savings b_{s,t}
                  (first column is zeros)
    Kpath_cnstr = (T+S-2,) Boolean vector, =True if K_t<=0
    Kpath       = (T+S-2,) vector, equilibrium time path of aggregate
                  capital stock K_t
    Y_params    = length 2 tuple, (A, alpha)
    Ypath       = (T+S-2,) vector, equilibrium time path of aggregate
                  output (GDP) Y_t
    Cpath       = (T+S-2,) vector, equilibrium time path of aggregate
                  consumption C_t
    RCerrPath   = (T+S-1,) vector, equilibrium time path of the resource
                  constraint error: Y_t - C_t - K_{t+1} + (1-delta)*K_t
    tpi_time    = scalar, time to compute TPI solution (seconds)
    tpi_output  = length 10 dictionary, {bpath, cpath, wpath, rpath,
                  Kpath, Ypath, Cpath, EulErrPath, RCerrPath, tpi_time}

    FILES CREATED BY THIS FUNCTION:
        Kpath.png
        Ypath.png
        Cpath.png
        wpath.png
        rpath.png
        bpath.png
        cpath.png

    RETURNS: tpi_output
    --------------------------------------------------------------------
    '''
    start_time = time.clock()
    (S, T, beta, sigma, chi, b_par, miu, l_endow, A, alpha, delta, b_ss, n_ss, K_ss, C_ss,
        L_ss, maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff) = params
    K1, K1_cnstr = c7ssf.get_K(bvec1)

    # Create time paths for K and L
    Kpath_init = np.zeros(T + S - 2)
    Kpath_init[:T] = get_path(K1, K_ss, T, "linear")
    Kpath_init[T:] = K_ss
    Lpath_init = L_ss * np.ones(T + S - 2)

    iter_TPI = int(0)
    dist_TPI = 10.
    Kpath_new = Kpath_init.copy()
    Lpath_new = Lpath_init.copy()
    r_params = (A, alpha, delta)
    w_params = (A, alpha)
    cbne_params = (S, T, beta, sigma,  chi, b_par, miu, l_endow, bvec1, b_ss, 
                   n_ss, TPI_tol, EulDiff)

    while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        Lpath_init = xi * Lpath_new + (1 - xi) * Lpath_init
        rpath = c7ssf.get_r(r_params, Kpath_init, Lpath_init)
        wpath = c7ssf.get_w(w_params, Kpath_init, Lpath_init)
        cpath, bpath, npath, EulErrPath_inter, EulErrPath_intra = get_cbnepath(cbne_params, rpath, wpath)
        Kpath_new = np.zeros(T + S - 2)
        Kpath_new[:T], Kpath_cnstr = c7ssf.get_K(bpath[:, :T])
        Kpath_new[T:] = K_ss * np.ones(S - 2)
        Lpath_new[:T - 1] = c7ssf.get_L(npath[:, :T - 1])
        Lpath_new[T - 1:] = L_ss * np.ones(S - 1)
        Kpath_cnstr = np.append(Kpath_cnstr,
                                np.zeros(S - 2, dtype=bool))
        Kpath_new[Kpath_cnstr] = 0.1
#        L_low = Lpath_new < 0
#        L_high = Lpath_new > 
#        Lpath_new[L_low] = 0.95 * S
#        Lpath_new[L_high] = 0.93 * S
        # Check the distance of Kpath_new1
        dist_TPI = ((Kpath_new[1:T] - Kpath_init[1:T]) ** 2).sum() + ((Lpath_new[:T] - Lpath_init[:T]) ** 2).sum()
        # dist_TPI = np.absolute((Kpath_new[1:T] - Kpath_init[1:T]) /
        #                        Kpath_init[1:T]).max()
        print('iter: ', iter_TPI, ', dist: ', dist_TPI,
              ',max Eul err Intra: ', np.absolute(EulErrPath_intra).max(),
              ',max Eul err Inter: ', np.absolute(EulErrPath_inter).max())

    if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
        print('TPI reached maxiter and did not converge.')
    elif iter_TPI == maxiter_TPI and dist_TPI <= mindist_TPI:
        print('TPI converged in the last iteration. ' +
              'Should probably increase maxiter_TPI.')
    Kpath = Kpath_new
    Lpath = Lpath_new
    Y_params = (A, alpha)
    Ypath = c7ssf.get_Y(Y_params, Kpath, Lpath)
    Cpath = np.zeros(T + S - 2)
    Cpath[:T - 1] = c7ssf.get_C(cpath[:, :T - 1])
    Cpath[T - 1:] = C_ss * np.ones(S - 1)
    RCerrPath = (Ypath[:-1] - Cpath[:-1] - Kpath[1:] +
                 (1 - delta) * Kpath[:-1])
    tpi_time = time.clock() - start_time

    tpi_output = {
        'bpath': bpath, 'cpath': cpath, 'npath': npath, 'wpath': wpath, 'rpath': rpath,
        'Kpath': Kpath, 'Lpath': Lpath, 'Ypath': Ypath, 'Cpath': Cpath,
        'EulErrPath IntraTemporal': EulErrPath_intra,'EulErrPath InterTemporal': EulErrPath_inter,
        'RCerrPath': RCerrPath,
        'tpi_time': tpi_time}

    # Print TPI computation time
    c7ssf.print_time(tpi_time, 'TPI')

    if graphs:
        '''
        ----------------------------------------------------------------
        cur_path    = string, path name of current directory
        output_fldr = string, folder in current path to save files
        output_dir  = string, total path of images folder
        output_path = string, path of file name of figure to be saved
        tvec        = (T+S-2,) vector, time period vector
        tgridTm1    = (T-1,) vector, time period vector to T-1
        tgridT      = (T,) vector, time period vector to T-1
        sgrid       = (S,) vector, all ages from 1 to S
        sgrid2      = (S-1,) vector, all ages from 2 to S
        tmatb       = (S-1, T) matrix, time periods for all savings
                      decisions ages (S-1) and time periods (T)
        smatb       = (S-1, T) matrix, ages for all savings decision
                      ages (S-1) and time periods (T)
        tmatc       = (3, T-1) matrix, time periods for all consumption
                      decisions ages (S) and time periods (T-1)
        smatc       = (3, T-1) matrix, ages for all consumption
                      decisions ages (S) and time periods (T-1)
        ----------------------------------------------------------------
        '''
        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T + S - 2 , T + S - 2)
        minorLocator = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Kpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate capital stock K')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate capital $K_{t}$')
        output_path = os.path.join(output_dir, "Kpath")
        plt.savefig(output_path)
        plt.show()
        
        plt.plot(tvec, Lpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate labor supply L')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate labor supply $L_{t}$')
        output_path = os.path.join(output_dir, "Lpath")
        plt.savefig(output_path)
        plt.show()

        # Plot time path of aggregate output (GDP)
        fig, ax = plt.subplots()
        plt.plot(tvec, Ypath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate output (GDP) Y')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $Y_{t}$')
        output_path = os.path.join(output_dir, "Ypath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of aggregate consumption
        fig, ax = plt.subplots()
        plt.plot(tvec, Cpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for aggregate consumption C')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate consumption $C_{t}$')
        output_path = os.path.join(output_dir, "C_aggr_path")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of real wage
        fig, ax = plt.subplots()
        plt.plot(tvec, wpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for real wage w')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real wage $w_{t}$')
        output_path = os.path.join(output_dir, "wpath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of real interest rate
        fig, ax = plt.subplots()
        plt.plot(tvec, rpath, marker='D')
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Time path for real interest rate r')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real interest rate $r_{t}$')
        output_path = os.path.join(output_dir, "rpath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of individual savings distribution
        tgridT = np.linspace(1, T, T)
        sgrid2 = np.linspace(2, S, S - 1)
        tmatb, smatb = np.meshgrid(tgridT, sgrid2)
        cmap_bp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = Axes3D(fig)
#        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual savings $b_{s,t}$')
        strideval = max(int(1), int(round(S / 10)))
        ax.plot_surface(tmatb, smatb, bpath[:, :T], rstride=strideval,
                        cstride=strideval, cmap=cmap_bp)
        output_path = os.path.join(output_dir, "bpath")
        plt.savefig(output_path)
        # plt.show()

        # Plot time path of individual consumption distribution
        tgridTm1 = np.linspace(1, T - 1, T - 1)
        sgrid = np.linspace(1, S, S)
        tmatc, smatc = np.meshgrid(tgridTm1, sgrid)
        cmap_cp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = Axes3D(fig)
#        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual consumption $c_{s,t}$')
        strideval = max(int(1), int(round(S / 10)))
        ax.plot_surface(tmatc, smatc, cpath[:, :T - 1],
                        rstride=strideval, cstride=strideval,
                        cmap=cmap_cp)
        output_path = os.path.join(output_dir, "cpath")
        plt.savefig(output_path)
        # plt.show()

    return tpi_output
    

def get_cbnepath(params, rpath, wpath):
    '''
    --------------------------------------------------------------------
    Generates matrices for the time path of the distribution of
    individual savings, individual consumption, and the Euler errors
    associated with the savings decisions.
    --------------------------------------------------------------------
    INPUTS:
    params  = length 9 tuple,
              (S, T, beta, sigma, nvec, bvec1, b_ss, TPI_tol, EulDiff)
    S       = integer in [3,80], number of periods an individual lives
    T       = integer > S, number of time periods until steady state
    beta    = scalar in (0,1), discount factor for each model period
    sigma   = scalar > 0, coefficient of relative risk aversion
    nvec    = (S,) vector, exogenous labor supply n_s
    bvec1   = (S-1,) vector, initial period savings distribution
    b_ss    = (S-1,) vector, steady-state savings distribution
    TPI_tol = scalar > 0, tolerance level for fsolve's in TPI
    EulDiff = Boolean, =True if want difference version of Euler errors
              beta*(1+r)*u'(c2) - u'(c1), =False if want ratio version
              [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    rpath   = (T+S-2,) vector, equilibrium time path of interest rate
    wpath   = (T+S-2,) vector, equilibrium time path of the real wage

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        paths_life()

    OBJECTS CREATED WITHIN FUNCTION:
    cpath       = (S, T+S-2) matrix, time path of the distribution of
                  consumption
    bpath       = (S-1, T+S-2) matrix, time path of the distribution of
                  savings
    EulErrPath  = (S-1, T+S-2) matrix, time path of Euler errors
    pl_params   = length 5 tuple, (S, beta, sigma, TPI_tol, EulDiff)
    p           = integer >= 2, index representing number of periods
                  remaining in a lifetime, used to solve incomplete
                  lifetimes
    b_guess     = (p-1,) vector, initial guess for remaining lifetime
                  savings, taken from previous cohort's choices
    bveclf      = (p-1,) vector, optimal remaining lifetime savings
                  decisions
    cveclf      = (p,) vector, optimal remaining lifetime consumption
                  decisions
    b_err_veclf = (p-1,) vector, Euler errors associated with
                  optimal remaining lifetime savings decisions
    DiagMaskb   = (p-1, p-1) Boolean identity matrix
    DiagMaskc   = (p, p) Boolean identity matrix

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cpath, bpath, EulErrPath
    --------------------------------------------------------------------
    '''
    S, T, beta, sigma,  chi, b_par, miu, l_endow, bvec1, b_ss, n_ss, TPI_tol, EulDiff = params
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
    def n_err_last(n, *args):
         params, b_last, w, r = args
         sigma, chi, b_par, miu, l_endow = params
         n_err_params = (chi, b_par, miu, l_endow)
         mu_n = c7ssf.get_n_mu(n_err_params, w, n)
         mu_c = w * (w * n + (1 + r) * b_last) ** (-sigma)
         return mu_n - mu_c
    b_last = bpath [-1, 0]
    w0 = wpath[0]
    r0 = rpath[0]
    n_err_last_params = (sigma, chi, b_par, miu, l_endow)
    n_last = opt.fsolve(n_err_last, x0 = n_ss[-1], args = (n_err_last_params, b_last, w0, r0), xtol = TPI_tol)
#    print(n_last)
    cpath[S - 1,  0] = w0 * n_last + (1 + r0) * b_last
    npath[-1, 0] = n_last
    print(cpath[-1, 0], w0, n_last, r0, b_last)
    pl_params = (S, beta, sigma, chi, b_par, miu, l_endow, TPI_tol, EulDiff)
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


def paths_life(params, beg_age, beg_wealth, rpath, wpath,
               b_init, n_init):
    '''
    --------------------------------------------------------------------
    Solve for the remaining lifetime savings decisions of an individual
    who enters the model at age beg_age, with corresponding initial
    wealth beg_wealth. Variable p is an integer in [2, S] representing
    the remaining periods of life.
    --------------------------------------------------------------------
    INPUTS:
    params     = length 5 tuple, (S, beta, sigma, TPI_tol, EulDiff)
    S          = integer in [3,80], number of periods an individual
                 lives
    beta       = scalar in (0,1), discount factor for each model
                 period
    sigma      = scalar > 0, coefficient of relative risk aversion
    TPI_tol    = scalar > 0, tolerance level for fsolve's in TPI
    EulDiff    = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want
                 ratio version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1
    beg_age    = integer in [1,S-1], beginning age of remaining life
    beg_wealth = scalar, beginning wealth at beginning age
    nvec       = (p,) vector, remaining exogenous labor supplies
    rpath      = (p,) vector, remaining lifetime interest rates
    wpath      = (p,) vector, remaining lifetime wages
    b_init     = (p-1,) vector, initial guess for remaining lifetime
                 savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        LfEulerSys()
        c7ssf.get_cvec()
        c7ssf.get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    p            = integer in [2,S], remaining periods in life
    b_guess      = (p-1,) vector, initial guess for lifetime savings
                   decisions
    eullf_objs   = length 7 tuple, (beta, sigma, beg_wealth, nvec,
                   rpath, wpath, EulDiff)
    bpath        = (p-1,) vector, optimal remaining lifetime savings
                   decisions
    cpath        = (p,) vector, optimal remaining lifetime
                   consumption decisions
    c_cnstr      = (p,) boolean vector, =True if c_p <= 0,
    b_err_params = length 2 tuple, (beta, sigma)
    b_err_vec    = (p-1,) vector, Euler errors associated with
                   optimal savings decisions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpath, cpath, b_err_vec
    --------------------------------------------------------------------
    '''
    S, beta, sigma, chi, b_par, miu, l_endow, TPI_tol, EulDiff = params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
#    if len(nvec) != p:
#        sys.exit("Beginning age and length of nvec do not match.")
    b_guess =   b_init
    n_guess = n_init
    bn_guess = np.append(b_guess, n_guess)
    eullf_objs = (beta, sigma, chi, b_par, miu, l_endow, beg_wealth, p, rpath, wpath, EulDiff)
    bnpath = opt.fsolve(LfEulerSys, bn_guess, args=(eullf_objs),
                       xtol=TPI_tol)
    bpath = bnpath[ : p - 1]
    npath = bnpath[p - 1: ]
    cpath, c_cnstr = c7ssf.get_cvec(rpath, wpath,
                                    np.append(beg_wealth, bpath), npath)
    # cpath, c_cnstr = get_cvec_lf(rpath, wpath, nvec,
    #                              np.append(beg_wealth, bpath))
    b_err_params = (beta, sigma)
    b_err_vec = c7ssf.get_b_errors(b_err_params, rpath[1:], cpath,
                                   c_cnstr, EulDiff)
    n_err_params = (sigma, chi, b_par, miu, l_endow)
    n_err_vec = c7ssf.get_n_errors(n_err_params, wpath, cpath, c_cnstr, npath)
    return bpath, npath, cpath, b_err_vec, n_err_vec
    
    
def LfEulerSys(bnvec, *args):
    '''
    --------------------------------------------------------------------
    Generates vector of all Euler errors for a given bvec, which errors
    characterize all optimal lifetime decisions, where p is an integer
    in [2, S] representing the remaining periods of life
    --------------------------------------------------------------------
    INPUTS:
    bvec       = (p-1,) vector, remaining lifetime savings decisions
                 where p is the number of remaining periods
    args       = length 7 tuple, (beta, sigma, beg_wealth, nvec, rpath,
                 wpath, EulDiff)
    beta       = scalar in [0,1), discount factor
    sigma      = scalar > 0, coefficient of relative risk aversion
    beg_wealth = scalar, wealth at the beginning of first age
    nvec       = (p,) vector, remaining exogenous labor supply
    rpath      = (p,) vector, interest rates over remaining life
    wpath      = (p,) vector, wages rates over remaining life
    EulDiff    = Boolean, =True if want difference version of Euler
                 errors beta*(1+r)*u'(c2) - u'(c1), =False if want ratio
                 version [beta*(1+r)*u'(c2)]/[u'(c1)] - 1

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        c7ssf.get_cvec()
        c7ssf.get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    bvec2        = (p,) vector, remaining savings including initial
                   savings
    cvec         = (p,) vector, remaining lifetime consumption
                   levels implied by bvec2
    c_cnstr      = (p,) Boolean vector, =True if c_{s,t}<=0
    b_err_params = length 2 tuple, (beta, sigma)
    b_err_vec    = (p-1,) vector, Euler errors from lifetime
                   consumption vector

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_err_vec
    --------------------------------------------------------------------
    '''
    beta, sigma, chi, b_par, miu, l_endow, beg_wealth, p, rpath, wpath, EulDiff = args
    bvec = bnvec[ : p - 1]
    nvec = bnvec[p - 1 : ]
#    if len(nvec) != 10:
#        print(len(nvec), len(bvec))
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_cnstr = c7ssf.get_cvec(rpath, wpath, bvec2, nvec)
    b_err_params = (beta, sigma)
    b_err_vec = c7ssf.get_b_errors(b_err_params, rpath[1:], cvec,
                                   c_cnstr, EulDiff)
    n_err_params = (sigma, chi, b_par, miu, l_endow)
    n_err_vec = c7ssf.get_n_errors(n_err_params, wpath, cvec, c_cnstr, nvec)
    err_vec = np.append(b_err_vec, n_err_vec)
    return err_vec
    
    
def get_path(x1, xT, T, spec):
    '''
    --------------------------------------------------------------------
    This function generates a path from point x1 to point xT such that
    that the path x is a linear or quadratic function of time t.

        linear:    x = d*t + e
        quadratic: x = a*t^2 + b*t + c

    The identifying assumptions for quadratic are the following:

        (1) x1 is the value at time t=0: x1 = c
        (2) xT is the value at time t=T-1: xT = a*(T-1)^2 + b*(T-1) + c
        (3) the slope of the path at t=T-1 is 0: 0 = 2*a*(T-1) + b
    --------------------------------------------------------------------
    INPUTS:
    x1 = scalar, initial value of the function x(t) at t=0
    xT = scalar, value of the function x(t) at t=T-1
    T  = integer >= 3, number of periods of the path
    spec = string, "linear" or "quadratic"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    cc    = scalar, constant coefficient in quadratic function
    bb    = scalar, coefficient on t in quadratic function
    aa    = scalar, coefficient on t^2 in quadratic function
    xpath = (T,) vector, parabolic xpath from x1 to xT

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: xpath
    --------------------------------------------------------------------
    '''
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) + cc

    return xpath