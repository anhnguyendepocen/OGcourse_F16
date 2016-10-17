# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import sys
import os

S = 80
alpha = 0.35
L = (2/3) * S + 0.2 * (1/3)* S
A = 1
delta = 0.05
beta = 0.96 ** (80/S)
sigma = 3
SS_tol = 1e-13


# nvec
nvec = np.zeros(S)
nvec[:int(round(2 * S / 3))] = 1
nvec[int(round(2 * S / 3)):] = 0.2

# K
def K(bvec_guess):
    K = np.sum(bvec_guess)
    return K

# w
def w(bvec_guess, alpha, A):
    w = (1 - alpha) * A * (K(bvec_guess) / L) ** alpha
    return w
    
# r
def r(bvec_guess, alpha, A, delta):
    r = alpha * (L / K(bvec_guess)) ** (1 - alpha) - delta
    return r
 
# set up c_s = [c_1, c_2, c_3]
def  c_s(S, alpha, A, delta, bvec_guess):
    '''
    Inputs:c
        params = alpha, A, , delta
        w = w(bvec, params)
        r = r(bvec, params)
        L = L(nvec)
        K = K(bvec)

    Return:
        c_s  = np.array([c_1, c_2, c_3])
    '''
 
    w0 = w(bvec_guess, alpha, A)
    r0 = r(bvec_guess, alpha, A, delta)
    bt = np.append([0], bvec_guess)
    bt1 = np.append(bvec_guess, [0])
    

    cvec = w0 * nvec[:1] + (1+r0) * bt - bt1
    return cvec

# Y
def  Y(bvec_guess, alpha, A):
    Y = A * K(bvec_guess) ** alpha * L ** (1 - alpha)
    return Y

# feasibility
def feasible(S, alpha, A, delta, bvec_guess):
    
    # compute K and c_s
    c = c_s(S, alpha, A, delta, bvec_guess)

    # check K
    K_cnstr = K(bvec_guess) <= 0
    if K_cnstr == True:
        feasible = False

    # check bvec
    # Set b_cnstr to all False
    b_cnstr = np.zeros(S-1, dtype=bool)

    for s in range(S):
        c_cnstr = c_s(S, alpha, A, delta, bvec_guess) <= 0
        
        # if the c1 is unfeasible, b1 unfeasible
        if c_cnstr[0] == True:
            b_cnstr[0] == True

        if c_cnstr[S-1] == True:
            b_cnstr[S-1] == True
            
        else: 
        	if s < S-2 and c_cnstr[s] == True:
	        	b_cnstr[s] = True
	        	b_cnstr[s+1] = True

    feasible = (b_cnstr, c_cnstr, K_cnstr)
    return feasible

# Q2

def get_EulErr(beta, sigma, alpha, A, delta, bvec_guess):
    '''
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings
    Inputs:
        params      = length 4 tuple, (beta, sigma, chi_b, bsp1)
        beta        = scalar in [0,1), discount factor
        sigma       = scalar > 0, coefficient of relative risk aversion
        b1          = scalar, last period savings (intentional bequests)
        r           = scalar > 0 or [p-1,] vector, interest rate or time
                      path of interest rates with the last value being 0
        cvec        = [p,] vector, distribution of consumption by age
                      c_p
        c_constr    = [p,] boolean vector, =True if c<=0 for given bvec
        bsp1_constr = [S,] boolean vector, last element =True if
                      b_{S+1}<=0
       
    Functions called: None
    Objects in function:
        mu_c         = [p-1,] vector, marginal utility of current
                       consumption
        mu_cp1       = [p-1,] vector, marginal utility of next period
                       consumption
        b_errors_dyn = [p-1,] vector, dynamic Euler errors
        b_errors_sta = scalar, static Euler error on intentional
                       bequests
        b_errors     = [p,] vector, Euler errors with errors = 0
                       characterizing optimal savings bvec
    Returns: b_errors
    '''
    

    w0 = w(bvec_guess, alpha, A)
    r0 = r(bvec_guess, alpha, A, delta)

    c = c_s(S, alpha, A, delta, bvec_guess)
    c_cnstr = c_s(S, alpha, A, delta, bvec_guess) <= 0
    c[c_cnstr] = 9999. 
    # print (c)


    mu_ct = c[:-1] ** (-sigma) # marginal utility at time t
    mu_ct1 = c[1:] ** (-sigma) # expected marginal utility at time t+1


    mu_ct[c_cnstr[0:-1]] = -9999
    mu_ct1[c_cnstr[1:]] = 9999
    # print(mu_ct)

    EulErr_ss = (beta * (1 + r0) * mu_ct1) - mu_ct
    return EulErr_ss

def SS_eqns(bvec_guess, *params):
    S, beta, sigma, L, A, alpha, delta, SS_tol = params
    
    KK = K(bvec_guess)
    YY = Y(bvec_guess, alpha, A)
    rr = r(bvec_guess, alpha, A, delta)
    ww = w(bvec_guess, alpha, A)
    bt = np.append([0,0], bvec_guess[:-1])
    bt1 = np.append([0], bvec_guess)
    c = c_s(S, alpha, A, delta, bvec_guess)
    
    eqns = np.zeros(S-1)
    for i in range(S-1):
        eqns[i] = c[i] ** (-sigma) - beta * (1+rr)* c[i+1] ** (-sigma)
    return eqns


def get_SS(params, bvec_guess, SS_graphs = True):
    start_time = time.clock()
    # set up inputs

    S, beta, sigma, L, A, alpha, delta, SS_tol = params
    KK = K(bvec_guess)
    YY = Y(bvec_guess, alpha, A)
    rr = r(bvec_guess, alpha, A, delta)
    ww = w(bvec_guess, alpha, A)
    bt = np.append([0,0], bvec_guess[:-1])
    bt1 = np.append([0], bvec_guess)
    c = c_s(S, alpha, A, delta, bvec_guess)
   
    EulErr_ss = get_EulErr(beta, sigma, alpha, A, delta, bvec_guess)

    b_ss = opt.fsolve(SS_eqns, bvec_guess, args=(params), xtol = 1e-14)
    
    # get RCerr
    for i in range(len(c)):
        c[i] = ww * nvec[i] + (1+rr) * bt[i] - bt1[i]
    C = sum(c)
    RCerr_ss = YY - C - delta * KK

    elapsed_time = time.clock() - start_time

    ss_output = {'b_ss': b_ss, 'c_ss': c, 'w_ss': ww, 'r_ss': rr, 'K_ss': KK, 'Y_ss': YY, 'EulErr_ss': EulErr_ss, 'RCerr_ss': RCerr_ss, 'ss_time': elapsed_time}


    # ploting graphs
    if SS_graphs == True:
        # period = np.arange(1, S+1)
        # cur_path = os.path.split(os.path.abspath(__file__))[0]
        # output_fldr = "images"
        # output_dir = os.path.join(cur_path, output_fldr)
        # if not os.access(output_dir, os.F_OK):
        #     os.makedirs(output_dir)
        # plt.plot(period, c, '^-', label = 'beta = '+format(beta)+ \
        #     ' Consumption')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.plot(period, np.concatenate((np.array([0]), b_ss)), 'o-', label= \
        #     'beta =' + format(beta) + ' Saving')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.grid(b=True, which='major', color='0.65', linestyle='-')
        # output_path = os.path.join(output_dir, "consumption")
        # plt.xlabel(r'Period $t$')
        # plt.ylabel(r'Level $b_t$ $c_t$')
        # plt.savefig(output_path)
        # plt.show()

        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        age_pers = np.arange(1, S+1)
        fig, ax = plt.subplots()
        plt.plot(age_pers, c, marker='D', linestyle=':', label='consumption')
        plt.plot(age_pers, b_ss, marker='o', linestyle='--', label='saving')

        # for the minor ticks, use no labels; default NullFormatter
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Figure1: Specifications for guess of K time path', fontsize=15)
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'')
        plt.legend(loc='upper right')
        output_path = os.path.join(output_dir, "consumption- savings")
        plt.savefig(output_path)
        plt.show()



    return ss_output


# TPI















