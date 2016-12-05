"""
Economic Policy Analysis with Overlapping Genration Models
Problem Set #6
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

import demographic_functions as dem
import functions_PS6 as ps6
import pandas as pd


########## QUESTION 1 #########

totpers = 80
graph = True

fert_rates = dem.get_fert(totpers, graph)

########## QUESTION 2 ##########

mort_rates, infmort_rate = dem.get_mort(totpers, graph)

########## QUESTION 3 ##########

imm_rates = dem.get_imm_resid(totpers, graph)

########## QUESTION 4 ##########

''' part a '''

num_periods = 100
graphs = False

fert_rates = dem.get_fert(num_periods, graphs)
mort_rates, infmort_rate = dem.get_mort(num_periods, graphs)
imm_rates = dem.get_imm_resid(num_periods, graphs)

var_names = ('age', 'year1', 'year2')   
df_pop = pd.read_csv('pop_data.csv', thousands=',', header=0, names=var_names)
pop = df_pop.as_matrix()

pop1 = pop[:,1]
pop2 = pop[:,2]

g = sum(pop2)/sum(pop1) - 1

Omega = np.diag(imm_rates)
# first line of Omega
Omega1 = fert_rates*(1 - infmort_rate)
# off-diagonal entries of Omega
Omega2 = np.diag(1 - mort_rates[:-1]) 
Omega3 = np.vstack((np.zeros([1,99]),Omega2))
Omega4 = np.hstack((Omega3,np.zeros([100,1])))

Omega[0,:] = Omega[0,:] + Omega1
Omega = Omega + Omega4

w,v = np.linalg.eig(Omega)

w_real = w[w.real == w] # find real eigenvalues
# growth rate is the only eigenvalue > 1
g_bar = w_real[np.where(w_real > 1)] - 1
g_bar = g_bar.real 
# population is the (normalized) eigenvector associated
omega_bar = v[:, np.where(w_real > 1)]
omega_bar = omega_bar.real
omega_bar = np.reshape(omega_bar,100)

age = np.linspace(1, 100, 100)     
# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath("__file__"))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Create plots
plt.figure()
plt.plot(age, omega_bar)
#minorLocator = MultipleLocator(1)
#ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')        
plt.title("Stationary Population Distribution")
plt.xlabel("Age")
plt.ylabel("Population") 
plt.ylim((0, .14))
output_path = os.path.join(output_dir, "stat_pop")
plt.savefig(output_path)
plt.show()

''' part b '''

omega_hat = pop2 / np.linalg.norm(pop2)
T = 200
S = 80
E = 20
pop_mat = np.zeros([S+E, T+S-2])
pop_mat[:,0] = omega_hat
for t in range(1,T):
    pop_mat[:,t] = np.dot(Omega,pop_mat[:,t-1]) / (1 + g_bar)
pop_mat[:,T:] = np.transpose(np.matlib.repmat(omega_bar,S-2,1))

pop_mat_S = pop_mat[20:, 20:]

''' part c '''

age_S = np.linspace(E, S+E, S)
fig, ax = plt.subplots()
plt.plot(age_S, pop_mat_S[:,0], label='$t = 1$')
plt.plot(age_S, pop_mat_S[:,9], label='$t = 10$')
plt.plot(age_S, pop_mat_S[:,29], label='$t = 30$')
plt.plot(age_S, pop_mat_S[:,T-1], label='$t = T$')
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Population Distribution, age $\geq S$', fontsize=20)
plt.xlabel(r'Age')
plt.ylabel(r'Population')
#plt.xlim((0, S + 1))
plt.ylim((0., .15 ))
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "stat_pop_T")
plt.savefig(output_path)
plt.show()

''' part d '''

g_mat = sum(pop_mat[20:, 20:])
g_tilde = g_mat[1:]/g_mat[:-1] - 1

x_var = np.linspace(1, g_tilde.size, g_tilde.size)
plt.figure()
plt.plot(x_var, g_tilde)
plt.grid(b=True, which='major', color='0.65', linestyle='-')        
plt.title("Population Growth Rates, age $\geq 20$")
plt.xlabel("Priod")
plt.ylabel("Growth Rate") 
plt.ylim((-.01, .01))
output_path = os.path.join(output_dir, "growth_rate_pop")
plt.savefig(output_path)
plt.show()

########## QUESTION 5 ##########

S = 80
nvec = 0.2*np.ones(S)
for i in range(int(np.round(2*S/3))):
    nvec[i] = 1.

beta = 0.96
delta = 0.05
sigma = 2.2
A = 1.
alpha = 0.35
g_y = 0.03
SS_tol = 1e-9
EulDiff = True

graphs = True
bvec_guess = 0.1*np.ones(S-1)

params = (beta, delta, sigma, A, alpha, nvec, g_y, mort_rates[20:], imm_rates[20:],\
          omega_bar[20:], g_bar, SS_tol, EulDiff)

ss_output = ps6.get_SS(params, bvec_guess, graphs)




