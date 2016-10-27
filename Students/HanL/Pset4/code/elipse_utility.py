# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:31:37 2016

@author: Luke
"""
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import sys
import os

def mu_diff(params_vec, *args):
    zeta, nvec = args
    b, miu = params_vec
    mu_cfe = -1 * nvec **(1 / zeta)
    mu_elip = -1 * b * (nvec) ** (miu - 1) * (1 - nvec ** miu) ** ((1 - miu) / miu)  
    mu_diff_abs = sum((mu_cfe - mu_elip) ** 2)
    return mu_diff_abs
    
def elipse(zeta):
    nvec = np.linspace(0.05, 0.95, 1000)
    b_init = 2
    miu_init = 1
    bnd_elip = ((1e-12, None), (1e-12, None))
    elip_init = np.array([b_init, miu_init])
    param = (zeta, nvec)
    result = opt.minimize(mu_diff, x0 = elip_init, args = (param), method = 'L-BFGS-B',
                          tol = 1e-12, bounds = bnd_elip)
    b, miu = result['x']
    return b, miu

zeta = 0.8
b, miu = elipse(zeta)
nvec = np.linspace(0.05, 0.95, 1500)
mdu_elip_vec = b * (nvec) ** (miu - 1) * (1 - nvec ** miu) ** ((1 - miu) / miu)  
mdu_cfe = nvec **(1 / zeta)
mu_elip_vec = mdu_elip_vec[::-1]
mu_cfe = mdu_cfe[::-1]



cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
fig, ax = plt.subplots()
minorLocator = MultipleLocator(1)
plt.plot(nvec, mdu_elip_vec, label= 'Ellipitical Utility')
plt.plot(nvec, mdu_cfe, label = 'CFE Utility Function')
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Approximation of Utility Function')
plt.xlabel(r'Labor Supply $t$')
plt.ylabel(r'Marginal DisUtility $r_{t}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, "Utility Function")
plt.savefig(output_path)

cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
fig, ax = plt.subplots()
minorLocator = MultipleLocator(1)
plt.plot(nvec, mu_elip_vec, label= 'Ellipitical Utility')
plt.plot(nvec, mu_cfe, label = 'CFE Utility Function')
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Approximation of Utility Function')
plt.xlabel(r'Leisure $t$')
plt.ylabel(r'Marginal DisUtility $r_{t}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, "Utility Function- Leisure")
plt.savefig(output_path)

