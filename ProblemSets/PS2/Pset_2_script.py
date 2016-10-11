# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:46:06 2016

@author: Luke
"""
import numpy as np
import Pset_2_func as p2
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import time


S = [1, 2, 3]
nvec = np.array([1.0, 1.0, 0.2])
A = 1
alpha = 0.35
delta = 0.6415
beta = 0.442
f_params = np.array([S, nvec, A, alpha, delta, beta])
sigma = 3
L = nvec.sum()
SS_tol =  math.e**(-12)
nvec=np.array([1, 1, 0.2])
params = np.array([S,  nvec, A, alpha, delta, beta, sigma, L, SS_tol])

bvec_guess1 = np.array([1.0, 1.2])
bvec_guess2 = np.array([0.06, -0.001])
bvec_guess3 = np.array([0.1, 0.1])
bvec_guess = bvec_guess3

SS_graphs = True
nonSS_graphs = True

maxper = 50



#############
# Problem 1##
#############
var_list = np.array([np.array(['b_2', 'b_3']),np.array(['c_1', 'c_2', 'c_3']),'K'])
f_1 = p2.feasible(f_params, bvec_guess1)
f_2 = p2.feasible(f_params, bvec_guess2)
f_3 = p2.feasible(f_params, bvec_guess3)

b_fail = np.array(['b_2', 'b_3'])[f_1[0]]
c_fail = np.array(['c_1', 'c_2', 'c3'])[f_1[1]]
k_fail = np.array(['K',])[f_1[2]]
q1_fail = np.concatenate((b_fail , c_fail , k_fail))
print ("For problem a), the varaible that falls out the constaint are", \
        format(q1_fail))

b_fail = np.array(['b_2', 'b_3'])[f_2[0]]
c_fail = np.array(['c_1', 'c_2', 'c3'])[f_2[1]]
k_fail = np.array(['K'])[f_2[2]]
q2_fail = np.concatenate((b_fail , c_fail , k_fail))
print ("For problem b), the varaible that falls out the constaint are", \
        format(q2_fail))
        
b_fail = np.array(['b_2', 'b_3'])[f_3[0]]
c_fail = np.array(['c_1', 'c_2', 'c3'])[f_3[1]]
k_fail = np.array(['K'])[f_3[2]]
q3_fail = np.concatenate((b_fail , c_fail , k_fail))
print ("For problem c), the varaible that falls out the constaint are", \
        format(q3_fail))
print()
        

    


#############
# Problem 2##
#############
# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

ss_output= p2.get_SS(params, bvec_guess, SS_graphs)
print("The solution for beta=0.55 is the following")
for i in ss_output:
    print(i,' : ' ,ss_output[i])
plt.title('Figure1: Saving and Consumption Distribution beta=0.442', fontsize=15)
output_path1 = os.path.join(output_dir, "Consumption-Capital SS",)
plt.savefig(output_path1, bbox_inches='tight')

# uncomment the follwoing line to see Figure1
# plt.show()

params[5] = 0.55
ss_output2 = p2.get_SS(params, bvec_guess, SS_graphs)
print('')
print("The solution for beta=0.55 is the following")
for i in ss_output2:
    print(i,' : ' ,ss_output2[i])
params[5] = beta
plt.title('Figure2: Saving and Consumption Distribution for two beta', fontsize=15)
output_path2 = os.path.join(output_dir, "Consumption-Capital SS Betas")
plt.savefig(output_path2, bbox_inches='tight')


#############
# Problem 3##
#############
b_ss = ss_output['b_ss']
K_ss = ss_output['K_ss']
kk_start = np.array([0.8 * b_ss[0], 1.1 * b_ss[1]])
K_start = kk_start.sum()

Kpath_lin = p2.get_path(K_start, K_ss, maxper, 'linear')
Kpath_qdr = p2.get_path(K_start, K_ss, maxper, 'quadratic')
eql = p2.non_ss (params, bvec_guess, maxper, 'quadratic')
Kpath_eql = eql[0]
w = eql[1]
r = eql[2]


age_pers = np.arange(1, maxper+1)
fig, ax = plt.subplots()
plt.plot(age_pers, Kpath_lin, marker='D', linestyle=':', label='Linear')
plt.plot(age_pers, Kpath_qdr, marker='o', linestyle='--', label='Quadratic')
plt.plot(age_pers, Kpath_eql, marker='x', linestyle='-', label='Equlibrium')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Figure3: Specifications for guess of K time path', fontsize=15)
plt.xlabel(r'Period $t$')
plt.ylabel(r'Aggregate Cap. Stock $K_t$')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "Kpath_init_comp")
plt.savefig(output_path)

fig, ax = plt.subplots()
plt.plot(age_pers, w, marker='D', linestyle=':', label='Wage Path')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Figure4: Specifications for Wage Path', fontsize=15)
plt.xlabel(r'Period $t$')
plt.ylabel(r'Price $K_t$')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "W_path")
plt.savefig(output_path)

fig, ax = plt.subplots()
plt.plot(age_pers, r, marker='x', linestyle='-', label='Rent Path')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Figure5: Specifications for Rent Path', fontsize=15)
plt.xlabel(r'Period $t$')
plt.ylabel(r'Rent $K_t$')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "R_path")
plt.savefig(output_path)


