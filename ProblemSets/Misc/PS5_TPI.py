# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import PS5_ssfunc as ssf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import sys
import os

def get_path(x1, xT, T, spec):
    
    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) + cc

    return xpath


def LfEulerSys(bvec, *args):
    
    beta, sigma, chi_b, zeta_s, BQ, beg_wealth, nvec, rpath, wpath = args
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_cnstr = ssf.get_cvec(rpath, wpath, zeta_s, BQ, bvec2, nvec)
    b_err_params = (beta, sigma, chi_b)
    b_err_vec = ssf.get_b_errors(b_err_params, rpath[1:], cvec, bvec2,
                                   c_cnstr)
    return b_err_vec


def paths_life(params, beg_age, beg_wealth, nvec, rpath, wpath,
               b_init):
    
    S, beta, sigma, chi_b, zeta_s, BQ, TPI_tol= params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
    if len(nvec) != p:
        sys.exit("Beginning age and length of nvec do not match.")
    b_guess =  1.01 * b_init
    eullf_objs = (beta, sigma, chi_b, zeta_s, BQ, beg_wealth, nvec, rpath, wpath)
    bpath = opt.fsolve(LfEulerSys, b_guess, args=(eullf_objs),
                       xtol=TPI_tol)
    cpath, c_cnstr = ssf.get_cvec(rpath, wpath, zeta_s, BQ,
                                    np.append(beg_wealth, bpath), nvec)
    # cpath, c_cnstr = get_cvec_lf(rpath, wpath, nvec,
    #                              np.append(beg_wealth, bpath))
    b_err_params = (beta, sigma, chi_b)
    b_err_vec = ssf.get_b_errors(b_err_params, rpath[1:], cpath, bpath,
                                   c_cnstr)
    return bpath, cpath, b_err_vec
    

def get_cbepath(params, rpath, wpath):
    
    S, T, beta, sigma, chi_b, zeta_s, nvec, bvec1, b_ss, TPI_tol= params
    cpath = np.zeros((S, T + S - 2))
    bpath = np.append(bvec1.reshape((S, 1)),
                      np.zeros((S, T + S - 2)), axis=1)
    EulErrPath = np.zeros((S, T + S - 1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    cpath[S - 1, 0] = ((1 + rpath[0]) * bvec1[S - 2] +\
                       wpath[0] * nvec[S - 1] + zeta_s[-1] * (1 + rpath[0]) \
                        * bpath[-1, -1]) * (chi_b / (1 + chi_b))
    bpath[S - 1, 1] = ((1 + rpath[0]) * bvec1[S - 2] +\
                       wpath[0] * nvec[S - 1] + zeta_s[-1] * (1 + rpath[0]) \
                       * bpath[-1, -1]) / (1 + chi_b)
    for p in range(2, S):
        b_guess = np.diagonal(bpath[S - p:, :p])  # the agent doesn't live in S + 1
        BQ = (1 + rpath[ : p]) * bpath[-1, : p]
        pl_params = (S, beta, sigma, chi_b, zeta_s, BQ, TPI_tol)
        bveclf, cveclf, b_err_veclf = paths_life(
            pl_params, S - p + 1, bvec1[S - p - 1], nvec[-p:],
            rpath[:p], wpath[:p], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        DiagMaskb = np.eye(p, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        bpath[S - p:, 1:p + 1] = DiagMaskb * bveclf + bpath[S - p:, 1:p + 1]
        cpath[S - p:, :p] = DiagMaskc * cveclf + cpath[S - p:, :p]
        EulErrPath[S - p:, 1:p + 1] = (DiagMaskb * b_err_veclf +
                                   EulErrPath[S - p:, 1:p + 1])
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    DiagMaskb = np.eye(S, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    for t in range(1, T):  # Go from periods 1 to T-1
        b_guess = np.diagonal(bpath[:, t - 1:t + S - 1])
        BQ = (1 + rpath[t - 1 : t + S - 1]) * bpath[-1,t - 1 : t + S - 1]
        pl_params = (S, beta, sigma, chi_b, zeta_s, BQ, TPI_tol)
        bveclf, cveclf, b_err_veclf = paths_life(
            pl_params, 1, 0, nvec, rpath[t - 1:t + S - 1],
            wpath[t - 1:t + S - 1], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        bpath[:, t:t + S] = (DiagMaskb * bveclf +
                                 bpath[:, t:t + S])
        cpath[:, t - 1:t + S - 1] = (DiagMaskc * cveclf +
                                     cpath[:, t - 1:t + S - 1])
        EulErrPath[:, t:t + S] = (DiagMaskb * b_err_veclf +
                                      EulErrPath[:, t:t + S])

    return cpath, bpath, EulErrPath
    


def get_TPI(params, bvec1, graphs):
    
    start_time = time.clock()
    (S, T, beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss,
        maxiter_TPI, mindist_TPI, xi, TPI_tol) = params

    K1, K1_cnstr = ssf.get_K(bvec1)

    # Create time paths for K and L
    Kpath_init = np.zeros(T + S - 2)
    Kpath_init[:T] = get_path(K1, K_ss, T, "linear")
    Kpath_init[T:] = K_ss
    Lpath = L * np.ones(T + S - 2)

    iter_TPI = int(0)
    dist_TPI = 10.
    Kpath_new = Kpath_init.copy()
    r_params = (A, alpha, delta)
    w_params = (A, alpha)
    cbe_params = (T, beta, sigma, chi_b, zeta_s, nvec, bvec1, b_ss, TPI_tol)

    while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        rpath = ssf.get_r(r_params, Kpath_init, Lpath)
        wpath = ssf.get_w(w_params, Kpath_init, Lpath)
        cpath, bpath, EulErrPath = get_cbepath(cbe_params, rpath, wpath)
        Kpath_new = np.zeros(T + S - 2)
        Kpath_new[:T], Kpath_cnstr = ssf.get_K(bpath[:, :T])
        Kpath_new[T:] = K_ss * np.ones(S - 2)
        Kpath_cnstr = np.append(Kpath_cnstr,
                                np.zeros(S - 2, dtype=bool))
        Kpath_new[Kpath_cnstr] = 0.1
        # Check the distance of Kpath_new1
        dist_TPI = ((Kpath_new[1:T] - Kpath_init[1:T]) ** 2).sum()
        # dist_TPI = np.absolute((Kpath_new[1:T] - Kpath_init[1:T]) /
        #                        Kpath_init[1:T]).max()
        print('iter: ', iter_TPI, ', dist: ', dist_TPI,
              ',max Eul err: ', np.absolute(EulErrPath).max())

    if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
        print('TPI reached maxiter and did not converge.')
    elif iter_TPI == maxiter_TPI and dist_TPI <= mindist_TPI:
        print('TPI converged in the last iteration. ' +
              'Should probably increase maxiter_TPI.')
    Kpath = Kpath_new
    Y_params = (A, alpha)
    Ypath = ssf.get_Y(Y_params, Kpath, Lpath)
    Cpath = np.zeros(T + S - 2)
    Cpath[:T - 1] = ssf.get_C(cpath[:, :T - 1])
    Cpath[T - 1:] = C_ss * np.ones(S - 1)
    RCerrPath = (Ypath[:-1] - Cpath[:-1] - Kpath[1:] +
                 (1 - delta) * Kpath[:-1])
    tpi_time = time.clock() - start_time

    tpi_output = {
        'bpath': bpath, 'cpath': cpath, 'wpath': wpath, 'rpath': rpath,
        'Kpath': Kpath, 'Ypath': Ypath, 'Cpath': Cpath,
        'EulErrPath': EulErrPath, 'RCerrPath': RCerrPath,
        'tpi_time': tpi_time}

    # Print TPI computation time
    ssf.print_time(tpi_time, 'TPI')

    if graphs:

        # Create directory if images directory does not already exist
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T + S - 2, T + S - 2)
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
        plt.show()

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
        plt.show()

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
        plt.show()

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
        plt.show()

        # Plot time path of individual savings distribution
        tgridT = np.linspace(1, T, T)
        sgrid2 = np.linspace(2, S + 1, S)
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
        plt.show()

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
        plt.show()

    return tpi_output
    



    
