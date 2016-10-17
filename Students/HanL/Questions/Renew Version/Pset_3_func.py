# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:49:22 2016

@author: Luke
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:44:12 2016

@author: Luke
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 18:39:07 2016

@author: Luke
"""
#import packages
import numpy as np
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt



#Build up the fuctions for w, r, K and L
def get_K(bvec):
    '''
    Input captial vector returning total capital that period
    
    Input: bvec: length 2 np.array
    
    Output: Total capital- float
    '''
    return sum(bvec)
    
def get_L(nvec):
    '''
    Input labor supply vector returning total labor that period
    
    Input: nvec: length 2 np.array
    
    Output: Total labor- float/np.array
    '''
    return sum(nvec)
    
    
def get_w(bvec, params):
    '''
    Input the saving vector and parameters, to get the wage of that period
    
    Input: bvec: saving vector np.array
            params: a set of parameter vectors
            
    Output: the wage of that period- float
    '''
    alpha = params[3]
    A = params[2]
    nvec = params[1]
    L = sum(nvec)
    K = sum(bvec)
    return (1 - alpha) * A * (K / L) **alpha

    
def get_r(bvec, params):
    '''
    Input the saving vector and parameters, to get the wage of that period
    
    Input: bvec: saving vector np.array
            params: a set of parameter vectors
    
    Output: bvec 
    '''
    alpha = params[3]
    A = params[2]
    delta = params[4]
    nvec = params[1]
    L = sum(nvec)
    K = sum(bvec)
    return alpha * A * (L / K)**(1 - alpha) - delta

    
def get_cvec(bvec, w, r, params):
    '''
    Take in saving vector, wage and rental rate, output an agent's cosumption
    vector
    Remember:
    c1 = w1                      - b2
    c2 = W2      + (1 + r2) * b2 - b3
    c3 = 0.2* w3 + (1 + r3) * b3
    
    Input: 
        bvec: saving vector [b2,b3]
        w: wage;        can be either float or an array
                        float--steady state case
                        array--TPI case
        r: rental rate; can be either float or an array
                        float--steady state case
                        array--TPI case
    
    Output:
        Consumption for each period. [c1, c2, c3]
        
    '''
    nvec = params[1]
    S = params[0]
    if (type(w) == np.float64):
        r = np.array([0,] + [r,] * (len(S) - 1))
        w = np.array([w,] * len(S))
    else:
        r = np.concatenate(([0,], r))
    n = len(w)
#    print (nvec)
    print ('wage vector', w)
#    print(bvec)
    cvec = np.array(nvec[-n : ]) * w + (1 + r) * np.concatenate(([0], bvec)) - np.append(bvec, [0])
    return cvec
        

def get_Y(bvec, params):
    '''
    Take in saving vector and parameter vector, output current period produciton
    
    Input: 
            bvec: saving vector [b2,b3]
            params: parameter vector
            
    Output:
            Single float point value for steady state production
    '''
    alpha = params[3]
    nvec = params[1]
    A = params[2]
    K = get_K(bvec)
    L = get_L(nvec)
    return A * K**(alpha) * L**(1-alpha)
    
def get_varss(bvec_guess, params):
    S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
    K = sum(bvec_guess)
    L = sum(nvec)
    w = (1 - alpha) * (K / L) **alpha
    r = alpha * (L / K) ** (1 - alpha) - delta
    cvec = get_cvec(bvec_guess, w, r, params)
    return K, L, w, r, cvec


#build up the feasilbe function
def feasible(f_params, bvec_guess):
    '''
    Given parameter and a saving vector guess, output an array indicating 
    whether certain variable is valid
    Criterion: c_i <= 0; b_i <= 0; K <= 0 return True
                
    Input: 
            f_params = f_params = np.array([S, nvec, A, alpha, delta, beta])
            bvec_guess = [b2, b3]
            
    Output:
            An array takes in the following form:
            [[boolean, boolean], [boolean, boolean, boolean], [boolean]]
            with the following order
            [b, c, K]
    '''
    #get K and consumption
    S = f_params[0]
    K = get_K(bvec_guess)
    w = get_w(bvec_guess, f_params)
    r= get_r(bvec_guess, f_params)
    consumption = get_cvec(bvec_guess, w, r, f_params)
    
    #create boolean for consumption and capital
    c_b = [consumption[i] <= 0 for i in range(len(consumption))]
    K_b = (K <= 0)
    
    #discuss the case for captial per period
    #first null the reuslt for savings to all False
    b_b = [False, ] * (len(S) - 1)
    for i in range(len(c_b)):
        if (i == 0):
            b_b[i] = c_b[i]
        elif (i == len(c_b) - 1):
            if (c_b[i] == True):
                b_b[i-1] = True
        else:
            if (c_b[i] == True):
                b_b[i-1], b_b[i] = (True, True)
        
    result = np.array([np.array(b_b), np.array(c_b), np.array([K_b])])
    return result
    
    
    
    
# write Euler Error function
# always take *args in order: params, w ,r 
def EulErrFunc(bvec, *args):
    '''
    Input saving vectors, parameter, wage and rental rate information, output 
    the euler error for the two euler equations
    
    Input:
        params:         parameters
                        STEADY STATE CASE
        w: wage;        can be either float or an array
                        float--steady state case
                        array--TPI case
        r: rental rate; can be either float or an array
                        float--steady state case
                        array--TPI case
    Output:
        error = [error1, error2]
    '''
    # if we are getting the euler error for SS state    
    # for steay state, only the parameter vector is passed in
    if (len(args) == 1):
        params = args[0]   
        S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
        K, L, w, r, cvec = get_varss(bvec, params)
    # for Non-SS state
    elif (len(args) == 3):
        w = args[0]
        r = args[1]
        params = args[2]
        S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
        cvec = get_cvec(bvec, w, r, params)
    else:
        w = args[0]
        r = args[1]
        w = np.append([0], w)
#        r = np.append([0], r)
        params = args[2]
        b_bar = args[3]
        S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
        cvec = get_cvec(np.append([b_bar], bvec), w, r, params)

    print(format('consumption'), cvec)
    print('wage',w)
    print('rent' , r)    
    #marginal utility
    c_cnstr = cvec <= 0
    cvec[c_cnstr] = 9999
#    print(cvec)
    MU1 = cvec[0:-1] ** (-sigma)
    MU1[c_cnstr[0:-1]] = 9999
#    print(MU1)
    MU2 = cvec[1:] **(-sigma)
    MU2[c_cnstr[1:]] = 9999
    
    #get the euler errors
    error = beta * (1 + r) * MU2 - MU1
    error = error[len(error) - len(bvec):]
#    error[c_cnstr[0:-1]] = 9999
#    error[c_cnstr[1:]] = 9999
#    error = np.array([abs(i)**0.5 for i in error]).sum()
#    print(error)
    return error
#    
    
    
#    
# write steady state function
def get_SS(params, bvec_guess, SS_graphs):
    '''
    Take in parameter vector, initial guess for saving vector and a grpah 
    boolean, return the steady state solutions and output the consumption and
    saving plot
    
    Input:
            parmas: parameter vector
            bvec_guess: initial guess for saving vector
            SS_graphs: a boolean; True-generate graph; False- don't generate graph
            
    Output:
            soluction dictionary with following format:
            {'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss, \
            'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss, 'EulErr_ss': EulErr_ss, \
            'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
            
    '''
    
    import time
    start_time = time.clock()

    x = opt.fsolve(EulErrFunc, bvec_guess, args=(params), xtol = math.e**(-20))
    
    
#    x = opt.minimize(EulErrFunc, bvec_guess, args=(params), method ='Nelder-Mead', tol = 1e-12, options={'disp': False, 'maxiter': None, 'xatol': 1e-14,  'maxfev': 50000, 'maxiter' : 50000})
#    print(x)
    b_ss = x
    EulErr_ss = EulErrFunc(b_ss, params)
    w_ss = get_w(b_ss, params)
    r_ss = get_r(b_ss, params)
    c_ss = get_cvec(b_ss, w_ss, r_ss, params)
    C_ss = c_ss.sum()
    K_ss = get_K(b_ss)
    Y_ss = get_Y(b_ss, params)
    RCerr_ss = Y_ss - C_ss.sum() - params[4] * K_ss
    ss_time = time.clock() - start_time
    
    if SS_graphs:
        period = params[0]
        plt.plot(period, c_ss, '^-', label = 'beta = '+format(params[5])+ \
        ' Consumption')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.plot(period, np.concatenate((np.array([0]), b_ss)), 'o-', label= \
        'beta =' + format(params[5]) + ' Saving')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Level $b_t$ $c_t$')
    return {'b_ss': b_ss, 'c_ss': c_ss, 'w_ss': w_ss, 'r_ss': r_ss, \
            'K_ss': K_ss, 'Y_ss': Y_ss, 'C_ss': C_ss, 'EulErr_ss': EulErr_ss, \
            'RCerr_ss': RCerr_ss, 'ss_time': ss_time}
            
            
#        
def norm(K1, K2):
    '''
    Given two capital time path, calculate the euclidean distance between the two
    capital apth
    
    Input:
            K1: [b_1, ......, b_maxper]
            K2: [b'_1, ......, b'_maxper]
    
    Output:
            A float representing the euclidean distance
    '''
    result = 0
    for i in range(len(K1)):
        result += (K1[i] - K2[i])**2
    return result**0.5
#
#
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
#
#    
def get_kkpath(bvec_1, bvec_guess, w, r, maxper, params):
    '''
    Given period one saving, an initial guess for saving, wage, rental rate
    iteration period and parameters, output all of the agents saving decision
    
    Input:
            bvec_1: first period saving [0.8*b2, 1.1*b3]
            bvec_guess: initial guess for iteration
            w: wage vector
            r: rental rate vector
            
    Output:
            generate the saving time path of different agents
            [[b_21, b_32], [b_22, b_33], [b_23, b_34]......, [b_2maxper, b3maxper+1]]
    '''
    S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
    # initialize the saving time path with zeores
    b_path = np.append(bvec_1.reshape((len(S) - 1, 1)),np.zeros((len(S) - 1, maxper + len(S) - 3)), axis = 1)
    for i in range(len(S) - 2):
    # i + 1 represents number of period left to live 
        Diagmask = np.eye(i + 1, dtype = bool)
        w_s = w[0 : i + 2]
        r_s = r[0 : i + 2]
#        w_s = np.append([0], w_s)
        b_guess = np.diagonal(b_path[len(S) - 2 - i:, : i + 1])
        b_start = b_path[-i - 2, 0]
        x = opt.fsolve(EulErrFunc, b_guess, args=(w_s, r_s, params, b_start), xtol = SS_tol)
        b_path[len(S) - 2 - i : , 1 : 2 + i] += Diagmask * x
    Diagmask = np.eye(len(S) - 1, dtype = bool)
    for i in range(0 , maxper - 1):
        w_s = w[i : i + len(S)]
        r_s = r[i + 1: i + len(S)]
        x = opt.fsolve(EulErrFunc, bvec_guess, args = (w_s, r_s, params), xtol = SS_tol)
        b_path[:, i + 1 : i + len(S)] += Diagmask * x
    return b_path[:, :maxper]


#def get_mat(bvec_1, kk_path, maxper):
#    b_mat = np.zeros((len(s) - 1, maxper + len(S) - 2))
    
    
    
    
def get_pathlife(bvec_1, bvec_guess, w, r, params):
    S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
    b_path = opt.fsolve(EulErrFunc, bvec_guess, args=(w, r, params), xtol = math.e**(-14))
    c_path = get_cvec(b_path, w, r, params)
    Eul_path = EulErrFunc(b_path, w, r, params)
    return b_path, c_path, Eul_path
    
#def non_ss(params, bvec_guess, maxper, path):
        

def non_ss (params, bvec_guess, maxper, path):
    '''
    Take in parameter vector, initial guess for capital vectors, number of 
    periods, linear or quadratic path, and a nonSS_graphs statement ,return 
    the total capital time path
    
    Input:
            params
            bvec_guess: initial guess for saving [b2, b3]
            maxper: an integer specifying number of periods of the economy
    
    Output:
            K_now: Total Capital TPI
    '''
    # get the steady state capital amount
    
    S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol = params
    ss_output = get_SS(params, bvec_guess, False)
    w_ss = ss_output['w_ss']
    r_ss = ss_output['r_ss']
    b_ss = ss_output['b_ss']
    K_end = b_ss.sum()
    # denote the first agent saving vector as kk_start
    bvec_1 = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])    
    K_start = bvec_1.sum()
    period = np.linspace(1, maxper, maxper)
    # get the initial TPI for capital by taking a linear path or quadratic path
    K_path = get_path(K_start, K_end, maxper, path)

    #initiallize the results
    K_past = K_path
    # take K_now as K_path minus one elementwise to create a large enough 
    # difference
    K_now = K_path - 1
    
    # keep track of the iteration to distinguish between the first iteratoin and
    # the rest
    iter = 0
    # set the criterion to 1e-10
    
    while (norm(K_past, K_now) >= 1e-10):
        # if we are solving the first iteration
        if iter == 0:
            # take the intital capital TPI as the input
            # have to equalize the vector or else the first period capital
            # would be the initial condition minus one
            K_now = K_past.copy()
            w = (1 - alpha) * (K_past / L) **alpha
            # wage and rental rate array has to have two additional period
            # so that the last agent can pin down his/her consumption
            w = np.concatenate((w, np.array([w_ss, ] * (len(S) - 1))))
            r = alpha * (L / K_past)**(1 - alpha) - delta
            r = np.concatenate((r, np.array([r_ss,] * (len(S) - 1))))

            b_mat = get_kkpath(bvec_1, bvec_guess, w ,r ,maxper, params)
            # for all of the agents' maximization problem, total capital equals
            # to the sum of the saving of the middle-aged and the young-aged
            K_now = np.sum(b_mat, axis = 0)
            # keep track of the iteration
            iter += 1
            
        else:
            K_temp = K_now   
            # let the input TPI be a linear conbination of the two capital TPI
            K_past = 0.3 * K_now.copy() + 0.7 * K_past.copy()
            w = (1 - alpha) * (K_past / L) **alpha
            w = np.concatenate((w, np.array([w_ss, ] * (len(S) - 2))))
            r = alpha * (L / K_past)**(1 - alpha) - delta
            r = np.concatenate((r, np.array([r_ss,] * (len(S) - 2))))

            b_mat = get_kkpath(bvec_1, bvec_guess, w , r, maxper, params)
            
            K_now = np.sum(b_mat, axis = 0)
            iter += 1
    print(b_mat)
    return [K_now, w[0: maxper], r[0: maxper]]
#            
            
                    
                
            

    
    
        

    








