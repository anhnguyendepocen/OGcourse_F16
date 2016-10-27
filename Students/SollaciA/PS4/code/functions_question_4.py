import time
import numpy as np
import scipy.optimize as opt
import functions_question_3 as fun3
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import sys
import os


def get_TPI(params, bvec1, graphs):

    start_time = time.clock()
    (S, T, beta, sigma, chi_n_vec, b_ellip, mu, l_tilde, A, alpha, delta, b_ss, n_ss, K_ss, C_ss,
        L_ss, maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff) = params
    K1, K1_cnstr = fun3.get_K(bvec1)

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
    cbne_params = (S, T, beta, sigma,  chi_n_vec, b_ellip, mu, l_tilde, bvec1, b_ss, 
                   n_ss, TPI_tol, EulDiff)

    while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        Lpath_init = xi * Lpath_new + (1 - xi) * Lpath_init
        rpath = fun3.get_r(r_params, Kpath_init, Lpath_init)
        wpath = fun3.get_w(w_params, Kpath_init, Lpath_init)
        cpath, bpath, npath, EulErrPath_inter, EulErrPath_intra = get_cbnepath(cbne_params, rpath, wpath)
        Kpath_new = np.zeros(T + S - 2)
        Kpath_new[:T], Kpath_cnstr = fun3.get_K(bpath[:, :T])
        Kpath_new[T:] = K_ss * np.ones(S - 2)
        Lpath_new[:T - 1] = fun3.get_L(npath[:, :T - 1])
        Lpath_new[T - 1:] = L_ss * np.ones(S - 1)
        Kpath_cnstr = np.append(Kpath_cnstr,
                                np.zeros(S - 2, dtype=bool))
        Kpath_new[Kpath_cnstr] = 0.1
        dist_TPI = ((Kpath_new[1:T] - Kpath_init[1:T]) ** 2).sum() + \
            ((Lpath_new[:T] - Lpath_init[:T]) ** 2).sum()

    if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
        print('TPI reached maximum interation but did not converge.')
        
    Kpath = Kpath_new
    Lpath = Lpath_new
    Y_params = (A, alpha)
    Ypath = fun3.get_Y(Y_params, Kpath, Lpath)
    Cpath = np.zeros(T + S - 2)
    Cpath[:T - 1] = fun3.get_C(cpath[:, :T - 1])
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
    fun3.print_time(tpi_time, 'TPI')

    if graphs:

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
   
def n_err_last(nvec, *args):
     params, b_last, w, r = args
     sigma, chi_n_vec, b_ellip, mu, l_tilde = params
     
     #n_err_params = (chi_n_vec, b_ellip, mu, l_tilde)
     mu_n = chi_n_vec * (b_ellip / l_tilde) * (nvec / l_tilde) ** (mu - 1) * (1 - (nvec / l_tilde) ** mu) ** ((1 - mu) / mu)

     mu_c = w * (w * nvec + (1 + r) * b_last) ** (-sigma)
     return mu_n - mu_c

def get_cbnepath(params, rpath, wpath):

    S, T, beta, sigma,  chi_n_vec, b_ellip, mu, l_tilde, bvec1, b_ss, n_ss, TPI_tol, EulDiff = params
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
    n_err_last_params = (sigma, chi_n_vec, b_ellip, mu, l_tilde)
    n_last = opt.fsolve(n_err_last, x0 = n_ss[-1], 
                        args = (n_err_last_params, b_last, w0, r0), xtol = TPI_tol)
#    print(n_last)
    cpath[S - 1,  0] = w0 * n_last + (1 + r0) * b_last
    npath[-1, 0] = n_last
    print(cpath[-1, 0], w0, n_last, r0, b_last)
    pl_params = (S, beta, sigma, chi_n_vec, b_ellip, mu, l_tilde, TPI_tol, EulDiff)
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


def paths_life(params, beg_age, beg_wealth, rpath, wpath,b_init, n_init):

    S, beta, sigma, chi_n_vec, b_ellip, mu, l_tilde, TPI_tol, EulDiff = params
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
    eullf_objs = (beta, sigma, chi_n_vec, b_ellip, mu, l_tilde, beg_wealth, p, rpath, 
                  wpath, EulDiff)
    bnpath = opt.fsolve(LfEulerSys, bn_guess, args=(eullf_objs),
                       xtol=TPI_tol)
    bpath = bnpath[ : p - 1]
    npath = bnpath[p - 1: ]
    cpath, c_cnstr = fun3.get_cvec(rpath, wpath,
                                    np.append(beg_wealth, bpath), npath)
    # cpath, c_cnstr = get_cvec_lf(rpath, wpath, nvec,
    #                              np.append(beg_wealth, bpath))
    b_err_params = (beta, sigma)
    b_err_vec = fun3.get_b_errors(b_err_params, rpath[1:], cpath,
                                   c_cnstr, EulDiff)
    n_err_params = (sigma, chi_n_vec, b_ellip, mu, l_tilde)
    n_err_vec = fun3.get_n_errors(n_err_params, wpath, cpath, c_cnstr, npath)
    return bpath, npath, cpath, b_err_vec, n_err_vec
    
    
def LfEulerSys(bnvec, *args):

    beta, sigma, chi_n_vec, b_ellip, mu, l_tilde, beg_wealth, p, rpath, wpath, EulDiff = args
    bvec = bnvec[:p-1]
    nvec = bnvec[p-1:]
#    if len(nvec) != 10:
#        print(len(nvec), len(bvec))
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_cnstr = fun3.get_cvec(rpath, wpath, bvec2, nvec)
    b_err_params = (beta, sigma)
    b_err_vec = fun3.get_b_errors(b_err_params, rpath[1:], cvec,
                                   c_cnstr, EulDiff)
    n_err_params = (sigma, chi_n_vec, b_ellip, mu, l_tilde)
    n_err_vec = fun3.get_n_errors(n_err_params, wpath, cvec, c_cnstr, nvec)
    err_vec = np.append(b_err_vec, n_err_vec)
    return err_vec
    
    
def get_path(x1, xT, T, spec):

    if spec == "linear":
        xpath = np.linspace(x1, xT, T)
    elif spec == "quadratic":
        cc = x1
        bb = 2 * (xT - x1) / (T - 1)
        aa = (x1 - xT) / ((T - 1) ** 2)
        xpath = aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) + cc

    return xpath