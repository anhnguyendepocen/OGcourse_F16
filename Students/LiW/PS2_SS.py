#import packages
import numpy as np
import scipy.optimize as opt
import math
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


'''
Check the feasibility of an initial guess for the steady-state with 
constraints of K > 0 and c_s > 0 for all s

Variables:
		K([b_2, b_3]) = aggregate capital stock
		r([b_2, b_3]) = interest rate
		w([b_2, b_3]) = real wage
		cvec        = consumption c_s for all s in S
		b_cnstr     = True if b_s >= 0
		K_cnstr     = True if K >= 0
		c_cnstr     = True if c >= 0
		feasibility = True if bvec is feasible

Returns: feasibility, K_cnstr, b_cnstr, c_cnstr
'''

# Inputs

S      = [1, 2, 3]
A      = 1                      # total factor productivity Cobb-Douglas production function
alpha  = 0.35                   # capital share of income in production function
delta  = 0.6415                 # 20-year depreciation rate of capital
beta   = 0.442					# 20-year discount factor
nvec   = np.array([1, 1, .2])   # exogenous labor supply n_s
L      = nvec.sum()             # aggregate labor        
f_params = (S, A, alpha, delta, beta, nvec)
sigma  = 3
SS_tol = math.e ** (-13)
params = np.array([S, nvec, A, alpha, delta, beta, sigma, L, SS_tol])


# K
def K(bvec):
	'''
	Inputs:
		bvec = np.array([b_2, b_3])

	Return:
		K(bvec) = b_2 + b_3
	'''
	K = np.sum(bvec)
	return K

# L
def L(nvec):
	'''
	Inputs:
		nvec = np.array([1, 1, .2])
	
	Retruns:
		L(nvec) = np.sum(nvec)
	'''
	L = np.sum(nvec)
	return L

# w
def w(bvec, params):
	'''
	Inputs:
		params = alpha, A
		L = L(nvec)
		K = K(bvec)

	Returns:
		w = (1 - alpha) * A * (K / L) ** alpha
	'''
	params = alpha, A
	w = (1 - alpha) * A * (K(bvec) / L(nvec)) ** alpha
	
	return w
	
# r
def r(bvec, nvec, params):
	'''
	Inputs:
		params = alpha, A, delta
		L = L(nvec)
		K = K(bvec)

	Return:
		r = alpha * (L / K) ** (1 - alpha) - delta
	'''

	params = alpha, A, delta
	
	return alpha * (L(nvec) / K(bvec)) ** (1 - alpha) - delta

# set up c_s = [c_1, c_2, c_3]
def  c_s(bvec, params):
	'''
	Inputs:
		params = alpha, A, L, delta
		w = w(bvec, params)
		r = r(bvec, params)
		L = L(nvec)
		K = K(bvec)

	Return:
		c_s  = np.array([c_1, c_2, c_3])
	'''
 
	nvec = np.array([1, 1, .2]) 
	params = alpha, A, delta
	w0 = w(bvec, params)
	r0 = r(bvec, nvec, params)
	b2 = bvec[0]
	b3 = bvec[1]
	
	cvec = np.zeros(3)
	cvec[0] = w0 - b2
	cvec[1] = w0 + (1 + r0) * b2 - b3
	cvec[2] = nvec[2] * w0 + (1 + r0) * b3

	return cvec

# Y
def  Y(params, bvec):
	'''
	Inputs:
		params = A, alpha
		K = K(bvec)
		L = L(bvec)

	Return:
		Y = A * K ** alpha * L ** (1 - alpha)
	'''
	
	params = A, alpha
	
	Y = A * K(bvec) ** alpha * L(bvec) ** (1 - alpha)

	return Y

# feasibility
def feasible(f_params, bvec_guess, nvec):


	# compute K and c_s

	c = c_s(bvec_guess, f_params)

	# check K
	K_cnstr = K(bvec_guess) <= 0
	if K_cnstr == True:
		feasible = False

	# check cvec
	c_cnstr = [c[0] <= 0, c[1] <=0, c[2] <= 0]

	# check bvec_guess
	# Set b_cnstr to all False
	b_cnstr = [False, False]
	# if c_1 <= 0, c_cnstr == Ture, b_cnstr[0] == True
	if c_cnstr[0] == True:
		b_cnstr[0] == True
	# if c_3 <= 0. c_cnstr == Ture, b_cnstr[1] == True
	if c_cnstr[2] == True:
		b_cnstr[1] == True
	# if c_2 <=0, c_cnstr == True, b_cnstr = [True, True]
	if c_cnstr[1] == True:
		b_cnstr = [True, True]

	feasible = (b_cnstr, c_cnstr, K_cnstr)
	return feasible

	print (feasible(f_params, np.array([1.0, 1.2]), nvec))
	print (feasible(f_params, np.array([0.06, -0.001]), nvec))
	print (feasible(f_params, np.array([0.1, 0.1]), nvec))

	
# Euler Error Function
	
def EulErr_ss(bvec, *args):
	'''
	args = params, w, r
	params = arg[0] = np.array([S, nvec, A, alpha, delta, beta, sigma, L, SS_tol])
	sigma = params[6]
	beta = params[5]
	K = K(bvec)
	L = L(nvec)
	w = w(bvec, params)
	r = r(bvec, params)
	'''

	f_params = S, nvec, A, alpha, delta, beta, sigma, L, SS_tol
	w0 = w(bvec, params)
	r0 = r(bvec, nvec, params)
	b_2 = bvec[0]
	b_3 = bvec[1]
	print(K(bvec), L(nvec), w0, r0)

	#marginal utilities

	mu1 = (w0 - b_2) ** (-sigma)
	mu2 = (w0 + (1 + r0) * b_2 - b_3) ** (-sigma)
	mu3 = (nvec[2] * w0 + (1 + r0) * b_3) ** (-sigma)

	print (mu1, mu2, mu3)

	# Euler Errors
	c = c_s(f_params, bvec)
	EulErr_ss = np.zeros(2)
	EulErr_ss[0] = beta * (1+r) * c[1] ** (-sigma) -  c[0] ** (-sigma)
	EulErr_ss[1] = beta * (1+r) * c[2] ** (-sigma) -  c[1] ** (-sigma)
	
	return EulErr_ss

def get_SS(params, bvec_guess, nvec, SS_graphs):
	start_time = time.clock()
	params = beta, sigma, L, A, alpha
	f_params = S, nvec, A, alpha, delta, beta, sigma, L, SS_tol

	# Values at steady states
	# savings
	b_ss = opt.fsolve(EulErr_ss, bvec_guess, args=(params), xtol = SS_tol)

	# aggregate capital
	K_ss = K(b_ss)

	# interest rate
	r_params = alpha, A, delta
	r_ss = r(bvec_guess, nvec, r_params)

	# real wages
	w_params = alpha, A
	w_ss = w(bvec_guess, w_params)

	# consumption
	c_params = alpha, A, delta
	c_ss = c_s(bvec_guess, c_params)

	# production
	Y_params = A, alpha
	Y_ss = Y(Y_params, bvec_guess)

	# resource constraint error
	RCerr_ss = Y_ss - c_ss.sum() - delta * K_ss

	elapsed_time = time.clock() - start_time

	print (get_SS(params, bvec_guess, nvec, SS_graphs)) 

	return (b_ss, c_ss, w_ss, r_ss, K_ss, Y_ss, EulErr_ss(b_ss, params), elapsed_time)

	
 

def print_time(seconds):
    '''
    --------------------------------------------------------------------
    Takes a total amount of time in seconds and prints it in terms of
    more readable units (days, hours, minutes, seconds)
    --------------------------------------------------------------------
    INPUTS:
    seconds = scalar > 0, total amount of seconds

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    secs = scalar > 0, remainder number of seconds
    mins = integer >= 1, remainder number of minutes
    hrs  = integer >= 1, remainder number of hours
    days = integer >= 1, number of days

    RETURNS: Nothing
    --------------------------------------------------------------------
    '''
    if seconds < 60:  # seconds
        secs = round(seconds, 4)
        print('SS computation time: ' + str(secs) + ' sec')
    elif seconds >= 60 and seconds < 3600:  # minutes
        mins = int(seconds / 60)
        secs = round(((seconds / 60) - mins) * 60, 1)
        print('SS computation time: ' + str(mins) + ' min, ' +
              str(secs) + ' sec')
    elif seconds >= 3600 and seconds < 86400:  # hours
        hrs = int(seconds / 3600)
        mins = int(((seconds / 3600) - hrs) * 60)
        secs = round(((seconds / 60) - hrs * 60 - mins) * 60, 1)
        print('SS computation time: ' + str(hrs) + ' hrs, ' +
              str(mins) + ' min, ' + str(secs) + ' sec')
    elif seconds >= 86400:  # days
        days = int(seconds / 86400)
        hrs = int(((seconds / 86400) - days) * 24)
        mins = int(((seconds / 3600) - days * 24 - hrs) * 60)
        secs = round(
            ((seconds / 60) - days * 24 * 60 - hrs * 60 - mins) * 60, 1)
        print('SS computation time: ' + str(days) + ' days, ' +
              str(hrs) + ' hrs, ' + str(mins) + ' min, ' +
              str(secs) + ' sec')



# TPI
params = input("What are params?")

get_SS(params, bvec_guess, nvec, SS_graphs)


