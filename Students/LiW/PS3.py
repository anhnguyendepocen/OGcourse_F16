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
A = 1
delta = 0.05
beta = 0.96 ** (80/S)
sigma = 3
SS_tol = 1e-13

# nvec
nvec = np.zeros(S)
nvec[:int(round(2 * S / 3))] = 1
nvec[int(round(2 * S / 3)):] = 0.2

# L
L = nvec.sum()

params = (S, beta, sigma, L, A, alpha, delta)

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
 
# set up c_s
def  c_s(S, alpha, A, delta, nvec, bvec_guess):
   
    w0 = w(bvec_guess, alpha, A)
    r0 = r(bvec_guess, alpha, A, delta)
    bt = np.append([0], bvec_guess)
    bt1 = np.append(bvec_guess, [0])
    

    cvec = w0 * nvec + (1+r0) * bt - bt1
    return cvec

# Y
def  Y(bvec_guess, alpha, A):
    Y = A * K(bvec_guess) ** alpha * L ** (1 - alpha)
    return Y

# feasibility
def feasible(S, alpha, A, delta, bvec_guess):
    
 
    c = c_s(S, alpha, A, delta, nvec, bvec_guess)

    K_cnstr = K(bvec_guess) <= 0
    if K_cnstr == True:
        feasible = False

    # check bvec
    # Set b_cnstr to all False
    b_cnstr = np.zeros(S-1, dtype=bool)

    for s in range(S):
        c_cnstr = c <= 0
        
        # if the c1 is unfeasible, b1 unfeasible
        if c_cnstr[0] == True:
            b_cnstr[0] == True

        if c_cnstr[S-1] == True:
            b_cnstr[S-1] == True
            
        else: 
        	if 0 < s < S-2 and c_cnstr[s] == True:
	        	b_cnstr[s] = True
	        	b_cnstr[s+1] = True
	
    feasible = (b_cnstr, c_cnstr, K_cnstr)
    return feasible

# Q2

def get_EulErr(beta, sigma, alpha, A, delta, bvec_guess):
    
    w0 = w(bvec_guess, alpha, A)
    r0 = r(bvec_guess, alpha, A, delta)

    c = c_s(S, alpha, A, delta, nvec, bvec_guess)
    c_cnstr = c <= 0
    c[c_cnstr] = 9999. 

    mu_ct = c[:-1] ** (-sigma) # marginal utility at time t
    mu_ct1 = c[1:] ** (-sigma) # expected marginal utility at time t+1


    mu_ct[c_cnstr[0:-1]] = -9999
    mu_ct1[c_cnstr[1:]] = 9999

    EulErr_ss = (beta * (1 + r0) * mu_ct1) - mu_ct
    return EulErr_ss

def SS_eqns(bvec_guess, *params):

    params = (S, beta, sigma, L, A, alpha, delta, SS_tol)
    KK = K(bvec_guess)
    YY = Y(bvec_guess, alpha, A)
    rr = r(bvec_guess, alpha, A, delta)
    ww = w(bvec_guess, alpha, A)
    bt = np.append([0], bvec_guess)
    bt1 = np.append(bvec_guess, [0])
    c = c_s(S, alpha, A, delta, nvec, bvec_guess)
    
    eqns = np.zeros(S-1)
    for i in range(S-1):
        eqns[i] = c[i] ** (-sigma) - beta * (1+rr)* c[i+1] ** (-sigma)
    return eqns


def get_SS(params, bvec_guess, nvec, SS_graphs = True):
    start_time = time.clock()

    params = (S, beta, sigma, L, A, alpha, delta)
    KK = K(bvec_guess)
    YY = Y(bvec_guess, alpha, A)
    rr = r(bvec_guess, alpha, A, delta)
    ww = w(bvec_guess, alpha, A)
    bt = np.append([0], bvec_guess)
    bt1 = np.append(bvec_guess, [0])
    c = c_s(S, alpha, A, delta, nvec, bvec_guess)
   
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

        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)

        period = np.arange(S)
        period2 = np.arange(S-1)
        fig, ax = plt.subplots()
        plt.plot(period, c, marker='D', linestyle=':', label='consumption')
        plt.plot(period2, b_ss, marker='o', linestyle='--', label='saving')

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
bvec_guess = np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           -0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
           0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

params = S, beta, sigma, nvec, L, A, alpha, delta, SS_tol
b_ss = opt.fsolve(SS_eqns, bvec_guess, args=(params), xtol = 1e-14)
KT = b_ss.sum()
bvec_1 = ((1.5 - 0.87) / 78 *(S-2) + 0.87) * b_ss
K1 = bvec_1.sum()
T = 30

# Kpath0
def get_Kpath0(K1, KT, T, spec):
	bvec_1 = ((1.5 - 0.87) / 78 *(S-2) + 0.87) * b_ss 
	K1 = bvec_1.sum()
	KT = b_ss.sum()

	if spec == "linear":
	    Kpath0 = np.linspace(K1, KT, T)
	elif spec == "quadratic":
	    cc = K1
	    bb = 2 * (KT - K1) / (T - 1)
	    aa = (K1 - KT) / ((T - 1) ** 2)
	    Kpath0 = aa * (np.arange(0, T) ** 2) + (bb * np.arange(0, T)) + cc
	
	return Kpath0

Kpath0 = get_Kpath0(K1,KT,T,"quadratic")
def wpath(alpha, A, L):
    wt = []
    for t in range(T):
        w = (1 - alpha) * A * (Kpath0[t] / L) ** alpha
        wt.append(w)
    return wt


# get rt
def rpath(alpha, A, L, delta):
    rt = []
    for t in range(T):
        r = alpha * A * (L / Kpath0[t]) ** (1 - alpha) - delta
        rt.append(r)
    return rt


# def get_kkpath(bvec_1, bvec_guess, T, params):
 	
#  	S, beta, sigma, nvec, L, A, alpha, delta, SS_tol = params
#  	w = wpath(alpha, A, L)
#  	r = rpath(alpha, A, L, delta) 

#  	# initialize the saving time path with zeores
#  	b_path = np.append(bvec_1.reshape((S - 1, 1)),np.zeros((S - 1, T + S - 3)), axis = 1)
	
#  	for i in range(2, S - 2):
#  	# i + 1 represents number of period left to live 
#  	    Diagmask = np.eye(i + 1, dtype = bool)
#  	    r_s = r[0 : i + 2]
#  	    w_s = np.append([0], w[0 : i + 2])
#  	    b_guess = np.diagonal(b_path[S - 2 - i:, : i + 1])
#  	    b_start = b_path[-i - 2, 0]
#  	    # x = opt.fsolve(get_EulErr, bvec_guess, args=(w, r, params, bvec_guess), xtol = SS_tol)
#  	    b_ss = opt.fsolve(SS_eqns, b_guess, args=(params), xtol = 1e-14)
#  	    b_path[S - 2 - i : , 1 : 2 + i] += Diagmask * b_ss
	
#  	Diagmask = np.eye(S - 1, dtype = bool)
#  	for i in range(0 , T - 1):
#  	    w_s = w[i : i + S]
#  	    r_s = r[i + 1: i + S]
#  	    # x = opt.fsolve(get_EulErr, bvec_guess, args = (w, r, params, bvec_guess), xtol = SS_tol)
#  	    b_ss = opt.fsolve(SS_eqns, bvec_guess, args=(params), xtol = 1e-14)
#  	    b_path[:, i + 1 : i + S] += Diagmask * b_ss
#  	return b_path[:, :T]
	

# def get_pathlife(bvec_1, bvec_guess, w, r, params):
#    # S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
#    # b_path = opt.fsolve(EulErr, bvec_guess, args=(w, r, params), xtol = 1e-13)
#    # c_path = get_cvec(b_path, w, r, params)
#    # Eul_path = EulErr(b_path, w, r, params)
#    # return b_path, c_path, Eul_path

# 	params = (S, T, beta, sigma, nvec, b_ss, SS_tol)
# 	cpath = np.zeros((S, T + S - 2))
# 	bpath = np.append(bvec_1.reshape((S - 1, 1)), np.zeros((S - 1, T + S - 3)), axis=1)
# 	EulErrPath = np.zeros((S - 1, T + S - 2))

# 	cpath[S - 1, 0] = ((1 + rpath[0]) * bvec_1[S - 2] + wpath[0] * nvec[S - 1])

def norm(K1, K2):
 	result = 0
 	for i in range(len(K1)):
 	    result += (K1[i] - K2[i])**2
 	return result**0.5


