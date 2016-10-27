import numpy as np

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
    
def get_L(nvec, l_tilde):
    
    L = nvec.sum()
    n_low = nvec <= 0
    n_high = nvec >= l_tilde

    return L, n_low, n_high    

def get_cvec(r, w, bvec, nvec):
    
    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    cvec = (1 + r) * b_s + w * nvec - b_sp1
    if cvec.min() <= 0:
        print('get_cvec() warning: distribution of savings and/or ' +
              'parameters created c<=0 for some agent(s)')
    c_cnstr = cvec <= 0

    return cvec, c_cnstr 

def get_r(params, K, L):
    
    A, alpha, delta = params
    r = alpha * A * ((L / K) ** (1 - alpha)) - delta

    return r
    
def get_w(params, K, L):
    
    A, alpha = params
    w = (1 - alpha) * A * ((K / L) ** alpha)

    return w
    

def feasible(f_params, nvec, bvec):

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