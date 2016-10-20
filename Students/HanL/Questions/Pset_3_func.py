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
    return np.array(bvec).sum()
    
def get_L(nvec):
    '''
    Input labor supply vector returning total labor that period
    
    Input: nvec: length 2 np.array
    
    Output: Total labor- float/np.array
    '''
    return np.array(nvec).sum()
    
    
def get_w(bvec, params):
    '''
    Input the saving vector and parameters, to get the wage of that period
    
    Input: bvec: saving vector np.array
            params: a set of parameter vectors
            
    Output: the wage of that period- float
    '''
    alpha = params[3]
    A = params[2]
    L = get_L(params[1])
    K = get_K( bvec )
    return (1 - alpha) * (K / L) **alpha

    
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
    L = get_L(params[1])
    K = get_K(bvec)
    return alpha * (L / K)**(1 - alpha) - delta

    
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
    # initialize the parameters
    nvec = params[1]
    S = params[0]    
    # steady state case will input a single float representing the w_ss
    if (type(w) == np.float64):
        w = np.array([w] * len(S)
        r = np.array([0,] + [r,] * (len(S) - 1))
    # for TPI case, the input is an array containg wage and rental information
    # for differrent period
    else:
        r = np.concatenate (([0], r))
    # represent cvec in a vector form according to the specification above
    cvec = np.array(nvec) * w + (np.ones(len(S)) + r) * np.concatenate(([0], bvec))\
            - np.concatenate(bvec, [0])
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
        if (i == len(s) - 1):
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
    if (len(args) == 1):
        params = args[0]
        sigma = params[6]
        beta = params[5]
        r = get_r(bvec, params)
        w = get_w(bvec, params)
        cvec = get_cvec(bvec, w, r)
    # for Non-SS state
    else:
        w = args[0]
        r = args[1]
        params = args[2]
        sigma = params[6]
        beta = params[5]
        cvec = get_cvec(bvec, w, r)
        
    # initialize consumption
    c1 = cvec[0]
    c2 = cvec[1]
    c3 = cvec[2]
    
    #get the marginal utility for period 1
    mu1 = c1**(-sigma)
    # marginal utility for period 2 for young agent
    mu2 = c2**(-sigma)
    #marginal utility for period 3
    MU1 = cvec[0:-1] ** (-sigma)
    MU2 = cvec[1:] **(-sigma)
    
#    mu3 = c3**(-sigma)
#    MU1 = np.array([mu1, mu2])
#    MU2 = np.array([mu2, mu3])
    #get the euler errors
    error = beta * (1 + r) * MU2 - MU1
    
    return error
    
    
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

    x = opt.fsolve(EulErrFunc, bvec_guess, args=(params), xtol = math.e**(-14))

    b_ss = x
    EulErr_ss = EulErrFunc(b_ss, params)
    w_ss = get_w(b_ss, params)
    r_ss = get_r(b_ss, params)
    c_ss = get_cvec(b_ss, w_ss, r_ss)
    C_ss = c_ss.sum()
    K_ss = get_K(b_ss)
    Y_ss = get_Y(b_ss, params)
    RCerr_ss = Y_ss - C_ss.sum() - params[4] * K_ss
    ss_time = time.clock() - start_time
    
    if SS_graphs:
        period = [1, 2, 3]
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
    alpha = params[3]
    sigma = params[-3]
    beta = params[5]
    # initialize the saving time path with zeors
    kk_path = [[0,0]]*maxper
    for i in range(maxper):
        # for the first agent, his/her young age saving is determined
        # thus he/she has a different maximization problem
        if i ==0:
            # young age saving is given
            b2 = bvec_1[0]
            w2 = w[0]
            w3 = w[1]
            r1 = r[0]
            r2 = r[1]
            #calculate middle age saving
            b3 = (w2 + (1 + r1) * b2 - 0.2 *((beta *(1 + r2))**\
            (-1/sigma)) * w3) / (1 + (1 + r2)*(beta *(1 + r2))**(-1/sigma))
            kk_path[i] = np.array([b2, b3])
        else:
            #for the rest of the agents
            w_t = np.array([w[i-1], w[i], w[i+1]])
            r_t = np.array([r[i], r[i+1]])
            x = opt.fsolve(EulErrFunc, bvec_guess, args=(w_t, r_t, params),\
                xtol = math.e**(-12))
            kk_path[i] = x
    return kk_path
    

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
    
    alpha = params[3]
    sigma = params[-3]
    beta = params[5]
    delta = params[4]
    L = params[7]
    ss_output = get_SS(params, bvec_guess, False)
    w_ss = ss_output['w_ss']
    r_ss = ss_output['r_ss']
    b_ss = ss_output['b_ss']
    K_end = b_ss.sum()
    # denote the first agent saving vector as kk_start
    kk_start = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])    
    K_start = kk_start.sum()
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
            w = np.concatenate((w, np.array([w_ss, w_ss])))
            r = alpha * (L / K_past)**(1 - alpha) - delta
            r = np.concatenate((r, np.array([r_ss, r_ss])))

            kk_path = get_kkpath(kk_start, bvec_guess, w ,r ,maxper, params)
            # for all of the agents' maximization problem, total capital equals
            # to the sum of the saving of the middle-aged and the young-aged
            for  i in range(len(kk_path)-1):
                K_now[i+1] = kk_path[i][1] + kk_path[i+1][0]
            # keep track of the iteration
            iter += 1
            
        else:
            K_temp = K_now   
            # let the input TPI be a linear conbination of the two capital TPI
            K_past = 0.3 * K_now.copy() + 0.7 * K_past.copy()
            w = (1 - alpha) * (K_past / L) **alpha
            w = np.concatenate((w, np.array([w_ss, w_ss])))
            r = alpha * (L / K_past)**(1 - alpha) - delta
            r = np.concatenate((r, np.array([r_ss, r_ss])))

            kk_path = get_kkpath(kk_start, bvec_guess, w , r, maxper, params)
            
            for  i in range(len(kk_path)-1):
                K_now[i+1] = kk_path[i][1] + kk_path[i+1][0]
            iter += 1
    return [K_now, w[0: maxper], r[0: maxper]]
            
            
                    
                
            

    
    
        

    








