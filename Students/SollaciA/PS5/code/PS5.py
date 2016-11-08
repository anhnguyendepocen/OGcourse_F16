"""
Economic Policy Analysis with Overlapping Genration Models
Problem Set #5
Alexandre B. Sollaci
The Univeristy of Chicago
Fall 2016
"""

#import time
import numpy as np
#import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
#import sys
import os

import zeta_func as zf
import PS5_functions as ps5
import pandas as pd


df_main = pd.read_stata('p13i6.dta')
df_summ = pd.read_stata('rscfp2013.dta')

networth = np.array(df_summ.networth)
wgt = np.array(df_summ.wgt)

bq_1 = np.array(df_main.X5804) * wgt
year_1 = np.array(df_main.X5805)
bq_2 = np.array(df_main.X5809) * wgt
year_2 = np.array(df_main.X5810)
bq_3 = np.array(df_main.X5814) * wgt
year_3 = np.array(df_main.X5815)
age_2013 = np.array(df_main.X8022)

zeta_s = zf.get_zeta(bq_1, bq_2, bq_3, year_1, year_2, year_3, age_2013, wgt)

age = np.arange(len(zeta_s)) + 21

cur_path = os.path.split(os.path.abspath("__file__"))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

fig = plt.figure()
plt.hist(zeta_s, bins='auto')
plt.title("Histogram of $\zeta_s$")
plt.xlabel("Value")
plt.ylabel("Frequency") 
output_path = os.path.join(output_dir, "zeta_s_hist")
plt.savefig(output_path)
plt.show()

fig = plt.figure()
plt.bar(age, zeta_s)  
plt.title("$\zeta_s$ as a function of age $s$")
plt.xlabel("Age")
plt.ylabel("$\zeta_s$") 
output_path = os.path.join(output_dir, "zeta_func")
plt.savefig(output_path)
plt.show()   

###### part b #####
   
bq_11 = bq_1 * ( networth <= np.percentile(networth, 25))
bq_12 = bq_1 * np.logical_and( networth > np.percentile(networth, 25) ,
                   networth <= np.percentile(networth, 50))
bq_13 = bq_1 * np.logical_and( networth > np.percentile(networth, 50) ,
                   networth <= np.percentile(networth, 75))
bq_14 = bq_1 * ( networth > np.percentile(networth, 75))

bq_21 = bq_2 * ( networth <= np.percentile(networth, 25))
bq_22 = bq_2 * np.logical_and( networth > np.percentile(networth, 25) ,
                   networth <= np.percentile(networth, 50))
bq_23 = bq_2 * np.logical_and( networth > np.percentile(networth, 50) ,
                   networth <= np.percentile(networth, 75))
bq_24 = bq_2 * ( networth > np.percentile(networth, 75))

bq_31 = bq_3 * ( networth <= np.percentile(networth, 25))
bq_32 = bq_3 * np.logical_and( networth > np.percentile(networth, 25) ,
                   networth <= np.percentile(networth, 50))
bq_33 = bq_3 * np.logical_and( networth > np.percentile(networth, 50) ,
                   networth <= np.percentile(networth, 75))
bq_34 = bq_3 * ( networth > np.percentile(networth, 75))

# compute marginal distribution for each quartile separately
zeta_1s = zf.get_zeta(bq_11, bq_21, bq_31, year_1, year_2, year_3, age_2013, wgt)
zeta_2s = zf.get_zeta(bq_12, bq_22, bq_32, year_1, year_2, year_3, age_2013, wgt)
zeta_3s = zf.get_zeta(bq_13, bq_23, bq_33, year_1, year_2, year_3, age_2013, wgt)
zeta_4s = zf.get_zeta(bq_14, bq_24, bq_34, year_1, year_2, year_3, age_2013, wgt)

# for the joint distribution, need the weight of each quartile
zeta_js = np.concatenate((zeta_1s, zeta_2s, 
                         zeta_3s, zeta_4s), axis = 0) / 4
zeta_js = np.reshape(zeta_js , (4, 80))

quartile = np.arange(4) + 1
#Axes3D.plot(age, quartile, zeta_js)

agemat, qmat = np.meshgrid(age, quartile)
cmap_bp = matplotlib.cm.get_cmap('Blues')
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel(r'age')
ax.set_ylabel(r'quartile of networth')
ax.set_zlabel(r'$\zeta_{j,s}$ as a function of age and income quartile')
strideval = max(int(1), int(round(8)))
ax.plot_surface(agemat, qmat, zeta_js, rstride=strideval,
                cstride=strideval, cmap=cmap_bp)
output_path = os.path.join(output_dir, "zeta_js")
plt.savefig(output_path)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nbins = 80
for c, z in zip(['r', 'g', 'b', 'y'], quartile):
    ys = zeta_js[z - 1, :]
    xs = age
    ax.bar(xs, ys, zs=z, zdir='y', color=c, ec=c, alpha=0.8)
ax.set_xlabel('Quartile of net worth')
ax.set_ylabel('Age')
ax.set_zlabel('$\zeta_{j,s}$')
output_path = os.path.join(output_dir, "zeta_js_hist")
plt.title("Histogram of $\zeta_{j,s}$")
plt.savefig(output_path)
plt.show()

############# QUESTION 2 #####################

S = 80
nvec = 0.2*np.ones(S)

for i in range(int(np.round(2*S/3))):
    nvec[i] = 1.

L = sum(nvec)
A = 1.
alpha = 0.35
delta = 0.05
beta = 0.96
sigma = 2.2
chi_b = 1.
SS_tol = 1e-9
EulDiff = True

bvec_guess = 0.1*np.ones(S)  
SS_params = (beta, sigma, nvec, L, A, alpha, delta, chi_b, zeta_s, SS_tol, EulDiff)

ss_output = ps5.get_SS(SS_params, bvec_guess)

############## QUESTION 3 ######################

b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
C_ss = ss_output['C_ss']
maxiter_TPI = 400
mindist_TPI = 1e-9
xi = 0.8
T = 300
TPI_tol = 1e-9

bvec1 = 0.8*b_ss

TPI_params = (S, T, beta, sigma, chi_b, zeta_s, nvec, L, A, alpha, delta, b_ss, K_ss, C_ss,
        maxiter_TPI, mindist_TPI, xi, TPI_tol, EulDiff) 

#tpi_output = ps5.get_TPI(TPI_params, bvec1, False) 
    
    
    
    
    
    
    
    
    