import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def get_K(barr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate capital stock K or time path of
    aggregate capital stock K_t
    --------------------------------------------------------------------
    INPUTS:
    barr = (S-1,) vector or (S-1, T+S-2) matrix, values for steady-state
           savings (b_2,b_3,...b_S) or time path of distribution of
           savings

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    K       = scalar or (T+S-2,) vector, steady-state aggregate capital
              stock or time path of aggregate capital stock
    K_cnstr = Boolean or (T+S-2) Boolean, =True if K <= 0 or if K_t <= 0

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: K, K_cnstr
    --------------------------------------------------------------------
    '''
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
    
def get_L(nvec, l_tilde):
    '''
    --------------------------------------------------------------------
    Solve for aggregate labor L
    --------------------------------------------------------------------
    INPUTS:
    nvec = (S,) vector, exogenous labor supply values (n_1,n_2,...n_S)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    L = scalar > 0, aggregate labor

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: L
    --------------------------------------------------------------------
    '''
    L = nvec.sum()
    n_low = nvec <= 0
    n_high = nvec >= l_tilde

    return L, n_low, n_high    

def get_cvec(r, w, bvec, nvec):
    '''
    --------------------------------------------------------------------
    Generates vector of lifetime steady-state consumptions given savings
    decisions, parameters, and the corresponding steady-state interest
    rate and wage.
    --------------------------------------------------------------------
    INPUTS:
    r    = scalar > 0 or (p,) vector, steady-state interest rate or p
           remaining periods of interest rates
    w    = scalar > 0 or (p,) vector, steady-state wage or p remaining
           periods of interest rates
    bvec = (p,) vector, steady-state distribution of savings b_{s+1}
    nvec = (p,) vector, exogenous labor supply n_{s} in remaining
           periods

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    b_s     = (p,) vector, b_init in first element and bvec in last p-1
              elements
    b_sp1   = (p,) vector, bvec in first p-1 elements and 0 in last
              element
    cvec    = (p,) vector, consumption by age c_s
    c_cnstr = (p,) Boolean vector, =True if c_s <= 0

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: cvec, c_cnstr
    --------------------------------------------------------------------
    '''
    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    cvec = (1 + r) * b_s + w * nvec - b_sp1
    if cvec.min() <= 0:
        print('get_cvec() warning: distribution of savings and/or ' +
              'parameters created c<=0 for some agent(s)')
    c_cnstr = cvec <= 0

    return cvec, c_cnstr 

def get_r(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state interest rate r or time path of interest
    rates r_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 3 tuple, (A, alpha, delta)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    r = scalar > 0 or (T+S-2) vector, steady-state interest rate or time
        path of interest rate

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: r
    --------------------------------------------------------------------
    '''
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r
    
def get_w(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state wage w or time path of wages w_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 2 tuple, (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    K      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             capital stock or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, steady-state aggregate
             labor or time path of aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    w = scalar > 0 or (T+S-2) vector, steady-state wage or time path of
        wage

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: w
    --------------------------------------------------------------------
    '''
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w

    
def get_C(carr):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate consumption C or time path of
    aggregate consumption C_t
    --------------------------------------------------------------------
    INPUTS:
    carr = (S,) vector or (S, T) matrix, distribution of consumption c_s
           in steady state or time path for the distribution of
           consumption

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    C = scalar > 0 or (T,) vector, aggregate consumption or time path of
        aggregate consumption

    Returns: C
    --------------------------------------------------------------------
    '''
    if carr.ndim == 1:
        C = carr.sum()
    elif carr.ndim == 2:
        C = carr.sum(axis=0)

    return C
    
def get_Y(params, K, L):
    '''
    --------------------------------------------------------------------
    Solve for steady-state aggregate output Y or time path of aggregate
    output Y_t
    --------------------------------------------------------------------
    INPUTS:
    params = length 2 tuple, production function parameters
             (A, alpha)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0,1), capital share of income
    K      = scalar > 0 or (T+S-2,) vector, aggregate capital stock
             or time path of the aggregate capital stock
    L      = scalar > 0 or (T+S-2,) vector, aggregate labor or time
             path of the aggregate labor

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    Y = scalar > 0 or (T+S-2,) vector, aggregate output (GDP) or
        time path of aggregate output (GDP)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Y
    --------------------------------------------------------------------
    '''
    A, alpha = params
    Y = A * (K ** alpha) * (L ** (1 - alpha))

    return Y

def feasible(f_params, nvec, bvec):
    '''
    --------------------------------------------------------------------
    Check whether a vector of steady-state savings is feasible in that
    it satisfies the nonnegativity constraints on consumption in every
    period c_s > 0 and that the aggregate capital stock is strictly
    positive K > 0
    --------------------------------------------------------------------
    INPUTS:
    params = length 4 tuple, (nvec, A, alpha, delta)
    nvec   = (S,) vector, exogenous labor supply values (n_1,n_2,...n_S)
    A      = scalar > 0, total factor productivity
    alpha  = scalar in (0, 1), capital share of income
    delta  = scalar in (0, 1), per period depreciation rate
    bvec   = (S-1,) vector, values of steady-state savings
             (b_2,b_3,...b_S)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_L()
        get_K()
        get_w()
        get_r()
        get_cvec()

    OBJECTS CREATED WITHIN FUNCTION:
    L        = scalar > 0, steady-state aggregate labor
    K        = scalar, steady-state aggregate capital stock
    K_cnstr  = Boolean, =True if K <= 0
    w_params = length 2 tuple, (A, alpha)
    w        = scalar, steady-state wage
    r_params = length 3 tuple, (A, alpha, delta)
    r        = scalar, steady-state interest rate
    bvec2    = (S,) vector, steady-state savings distribution plus
               initial period wealth of zero
    cvec     = (S,) vector, steady-state consumption by age
    c_cnstr  = (S,) Boolean vector, =True for elements for which c_s<=0
    b_cnstr  = (S-1,) Boolean, =True for elements for which b_s causes a
               violation of the nonnegative consumption constraint

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_cnstr, c_cnstr, K_cnstr
    --------------------------------------------------------------------
    '''
    l_tilde, A, alpha, delta = f_params
    L, n_low, n_high = get_L(nvec, l_tilde)
    K, K_cnstr = get_K(bvec)
    w_params = (A, alpha)
    w = get_w(w_params, K, L)
    r_params = (A, alpha, delta)
    r = get_r(r_params, K, L)
    bvec2 = np.append([0], bvec)
    cvec, c_cnstr = get_cvec(r, w, bvec2, nvec)
    b_cnstr = c_cnstr[:-1] + c_cnstr[1:]

    return n_low, n_high, b_cnstr, c_cnstr, K_cnstr


def get_b_errors(params, r, cvec, c_cnstr, diff):
    '''
    --------------------------------------------------------------------
    Generates vector of dynamic Euler errors that characterize the
    optimal lifetime savings decision. Because this function is used for
    solving for lifetime decisions in both the steady-state and in the
    transition path, lifetimes will be of varying length. Lifetimes in
    the steady-state will be S periods. Lifetimes in the transition path
    will be p in [2, S] periods
    --------------------------------------------------------------------
    INPUTS:
    params  = length 2 tuple, (beta, sigma)
    beta    = scalar in (0,1), discount factor
    sigma   = scalar > 0, coefficient of relative risk aversion
    r       = scalar > 0 or (p,) vector, steady-state interest rate or
              time path of interest rates
    cvec    = (p,) vector, distribution of consumption by age c_p
    c_cnstr = (p,) Boolean vector, =True if c<=0 for given bvec
    diff    = boolean, =True if use simple difference Euler
              errors. Use percent difference errors otherwise.

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mu_c     = (p-1,) vector, marginal utility of current consumption
    mu_cp1   = (p-1,) vector, marginal utility of next period consumpt'n
    b_errors = (p-1,) vector, Euler errors characterizing optimal
               savings bvec

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
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

def get_n_errors(params, w, cvec, c_cnstr, nvec, n_low, n_high):
    b, mu, sigma, l_tilde = params
    nvec[n_low] = 0.9*l_tilde
    nvec[n_high] = 0.9*l_tilde
    cvec[c_cnstr] = 9999.

    mu_c = cvec ** (-sigma)
    mu_n = (b/l_tilde)*(nvec/l_tilde)**(mu-1)*(1 - (nvec/l_tilde)**(mu))**(1/mu - 1)
    n_errors = w*mu_c - mu_n
    
    return n_errors
    
def EulerSys(bn_vec, *args):
    '''
    --------------------------------------------------------------------
    Generates vector of all Euler errors that characterize optimal
    lifetime decisions
    --------------------------------------------------------------------
    INPUTS:
    bvec    = (S-1,) vector, distribution of savings b_{s+1}
    args    = length 8 tuple,
              (beta, sigma, nvec, L, A, alpha, delta, EulDiff)
    beta    = scalar in [0,1), discount factor
    sigma   = scalar > 0, coefficient of relative risk aversion
    nvec    = (S,) vector, exogenous labor supply n_{s}
    L       = scalar > 0, exogenous aggregate labor
    A       = scalar > 0, total factor productivity parameter in firms'
              production function
    alpha   = scalar in (0,1), capital share of income
    delta   = scalar in [0,1], model-period depreciation rate of capital
    EulDiff = Boolean, =True if want difference version of Euler errors
              beta*(1+r)*u'(c2) - u'(c1), =False if want ratio version
              [beta*(1+r)*u'(c2)]/[u'(c1)] - 1

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        get_K()
        get_r()
        get_w()
        get_cvec()
        get_b_errors()

    OBJECTS CREATED WITHIN FUNCTION:
    K            = scalar > 0, aggregate capital stock
    K_constr     = Boolean, =True if K<=0 for given bvec
    b_err_vec    = (S-1,) vector, vector of Euler errors
    r_params     = length 3 tuple, (A, alpha, delta)
    r            = scalar > 0, interest rate
    w_params     = length 2 tuple, (A, alpha)
    w            = scalar > 0, wage
    bvec2        = (S,) vector, steady-state savings distribution plus
                   initial period wealth of zero
    cvec         = (S,) vector, consumption c_s for each age-s agent
    c_constr     = (S,) Boolean vector, =True if c<=0 for given bvec
    b_err_params = length 2 tuple, (beta, sigma)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: b_errors
    --------------------------------------------------------------------
    '''
    S, beta, sigma, l_tilde, b, mu, A, alpha, delta, EulDiff = args
    bvec = bn_vec[:S-1]
    nvec = bn_vec[S-1:]
    K, K_cnstr = get_K(bvec)
    L, n_low, n_high = get_L(nvec, l_tilde)
    if K_cnstr:
        b_err_vec = 1000. * np.ones(nvec.shape[0] - 1)
        n_err_vec = 1000. * np.ones(nvec.shape[0])
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
        
        n_err_params = b, mu, sigma, l_tilde
        n_err_vec = get_n_errors(n_err_params, w, cvec, c_cnstr,
                                 nvec, n_low, n_high)
        
        err_vec = np.append(b_err_vec, n_err_vec)
        
    return err_vec


def get_cbepath(params, rpath, wpath):
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
    S, T, beta, sigma, nvec, bvec1, b_ss, TPI_tol, EulDiff = params
    cpath = np.zeros((S, T + S - 2))
    bpath = np.append(bvec1.reshape((S - 1, 1)),
                      np.zeros((S - 1, T + S - 3)), axis=1)
    EulErrPath = np.zeros((S - 1, T + S - 2))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    cpath[S - 1, 0] = ((1 + rpath[0]) * bvec1[S - 2] +
                       wpath[0] * nvec[S - 1])
    pl_params = (S, beta, sigma, TPI_tol, EulDiff)
    for p in range(2, S):
        b_guess = np.diagonal(bpath[S - p:, :p - 1])
        bveclf, cveclf, b_err_veclf = paths_life(
            pl_params, S - p + 1, bvec1[S - p - 1], nvec[-p:],
            rpath[:p], wpath[:p], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        DiagMaskb = np.eye(p - 1, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        bpath[S - p:, 1:p] = DiagMaskb * bveclf + bpath[S - p:, 1:p]
        cpath[S - p:, :p] = DiagMaskc * cveclf + cpath[S - p:, :p]
        EulErrPath[S - p:, 1:p] = (DiagMaskb * b_err_veclf +
                                   EulErrPath[S - p:, 1:p])
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    DiagMaskb = np.eye(S - 1, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    for t in range(1, T):  # Go from periods 1 to T-1
        b_guess = np.diagonal(bpath[:, t - 1:t + S - 2])
        bveclf, cveclf, b_err_veclf = paths_life(
            pl_params, 1, 0, nvec, rpath[t - 1:t + S - 1],
            wpath[t - 1:t + S - 1], b_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        bpath[:, t:t + S - 1] = (DiagMaskb * bveclf +
                                 bpath[:, t:t + S - 1])
        cpath[:, t - 1:t + S - 1] = (DiagMaskc * cveclf +
                                     cpath[:, t - 1:t + S - 1])
        EulErrPath[:, t:t + S - 1] = (DiagMaskb * b_err_veclf +
                                      EulErrPath[:, t:t + S - 1])

    return cpath, bpath, EulErrPath


def paths_life(params, beg_age, beg_wealth, nvec, rpath, wpath,
               b_init):
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
        get_cvec()
        get_b_errors()

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
    S, beta, sigma, TPI_tol, EulDiff = params
    p = int(S - beg_age + 1)
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
    if len(nvec) != p:
        sys.exit("Beginning age and length of nvec do not match.")
    eullf_objs = (beta, sigma, beg_wealth, nvec, rpath, wpath, EulDiff)
###############################################################################    
    bn_guess = np.append( 1.01 * b_init , nvec)
    bnpath = opt.fsolve(LfEulerSys, bn_guess, args=(eullf_objs),
                       xtol=TPI_tol)
    x = int( ( len(bnpath) + 1 ) / 2 )
    bpath = bnpath[:x-1]
    npath = bnpath[x-1:]
###############################################################################
    cpath, c_cnstr = get_cvec(rpath, wpath,
                                    np.append(beg_wealth, bpath), nvec)
    # cpath, c_cnstr = get_cvec_lf(rpath, wpath, nvec,
    #                              np.append(beg_wealth, bpath))
    b_err_params = (beta, sigma)
    b_err_vec = get_b_errors(b_err_params, rpath[1:], cpath,
                                   c_cnstr, EulDiff)
    return bpath, cpath, b_err_vec

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
        get_cvec()
        get_b_errors()

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
    beta, sigma, beg_wealth, nvec, rpath, wpath, EulDiff = args
    x = int( ( len(bnvec) + 1 ) / 2 )
    bvec = bnvec[:x-1]
    nvec = bnvec[x-1:]
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_cnstr = get_cvec(rpath, wpath, bvec2, nvec)
    b_err_params = (beta, sigma)
    b_err_vec = get_b_errors(b_err_params, rpath[1:], cvec,
                                   c_cnstr, EulDiff)
###############################################################################    
    L, n_low, n_high = get_L(nvec, l_tilde = 1)
    #n_err_params = b, mu, sigma, l_tilde
    n_err_params = .5, 1.5, sigma, 1
    n_err_vec = get_n_errors(n_err_params, wpath, cvec, c_cnstr,
                             nvec, n_low, n_high)
    #n_err_vec = np.zeros(len(n_err_vec))
    err_vec = np.append(b_err_vec, n_err_vec)
###############################################################################
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


    
def get_TPI(params, bvec1, nvec1, graphs):
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
        get_K()
        get_path()
        get_r()
        get_w()
        get_cbepath()
        get_Y()
        get_C()
        print_time()


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
    (S, T, beta, sigma, A, alpha, delta, b, mu, l_tilde, b_ss, K_ss, C_ss,
        maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff) = params
    K1, K1_cnstr = get_K(bvec1)
    L , n_low, n_high = get_L(nvec1, l_tilde)

    # Create time paths for K and L
    Kpath_init = np.zeros(T + S - 2)
    Kpath_init[:T] = get_path(K1, K_ss, T, "quadratic")
    Kpath_init[T:] = K_ss
    Lpath = L * np.ones(T + S - 2)

    iter_TPI = int(0)
    dist_TPI = 10.
    Kpath_new = Kpath_init.copy()
    r_params = (A, alpha, delta)
    w_params = (A, alpha)
    cbe_params = (S, T, beta, sigma, nvec1, bvec1, b_ss, TPI_tol,
                  EulDiff)

    while (iter_TPI < maxiter_TPI) and (dist_TPI >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        rpath = get_r(r_params, Kpath_init, Lpath)
        wpath = get_w(w_params, Kpath_init, Lpath)
        cpath, bpath, EulErrPath = get_cbepath(cbe_params, rpath, wpath)
        Kpath_new = np.zeros(T + S - 2)
        Kpath_new[:T], Kpath_cnstr = get_K(bpath[:, :T])
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
    Ypath = get_Y(Y_params, Kpath, Lpath)
    Cpath = np.zeros(T + S - 2)
    Cpath[:T - 1] = get_C(cpath[:, :T - 1])
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
    print_time(tpi_time, 'TPI')

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
        cur_path = os.path.split(os.path.abspath("__file__"))[0]
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
        # plt.show()

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
        ax = fig.gca(projection='3d')
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
        ax = fig.gca(projection='3d')
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

   
def print_time(seconds, type):
    '''
    --------------------------------------------------------------------
    Takes a total amount of time in seconds and prints it in terms of
    more readable units (days, hours, minutes, seconds)
    --------------------------------------------------------------------
    INPUTS:
    seconds = scalar > 0, total amount of seconds
    type = string, either "SS" or "TPI"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:

    OBJECTS CREATED WITHIN FUNCTION:
    secs = scalar > 0, remainder number of seconds
    mins = integer >= 1, remainder number of minutes
    hrs  = integer >= 1, remainder number of hours
    days = integer >= 1, number of days

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: Nothing
    --------------------------------------------------------------------
    '''
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
    

