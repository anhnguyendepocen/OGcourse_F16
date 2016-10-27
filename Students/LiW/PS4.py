# Import packages
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import PS4_script as func
import os
import time


# Parameters
S = 10
l_tilde = 1
A = 1
alpha = 0.35
delta = 0.05
beta = 0.96
sigma = 3
b_init = 2
mu_init = 1
chi = 1
theta = 0.8
SS_tol = 1e-12
ELP_init = np.array([b_init, mu_init])
nvec = np.linspace(0.05, 0.95, 1000)
ELP_args = (nvec, b_init, mu_init)
bnds_ELP = ((1e-12, None), (1e-12, None))
graphs = True


# Q1
ELP_params = opt.minimize(func.MU_n_sse, ELP_init, args=(ELP_args),
                          method='L-BFGS-B', bounds=bnds_ELP)

b, mu = ELP_params["x"]
print("1.(a)" "b = ", b, "mu = ", mu)

MU_ELP = (b / mu) * ((1- nvec) ** mu) ** ((1 - mu)/ mu)
MU_CFE = chi * (nvec) ** (1 / theta)

# Plot
period = np.linspace(0.05,0.95,1000)
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


plt.plot(period, MU_ELP)
plt.plot(period, MU_CFE)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('ELP - CFE', fontsize=20)
plt.xlabel(r'leisure')
plt.ylabel(r'utility of leisure')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, "ELP - CFE")
plt.savefig(output_path)
plt.show()


# Q2
f_params = l_tilde, A, alpha, delta

print("2.(a) :")
nvec_guess1 = 0.95 * np.ones(S)
bvec_guess1 = np.ones(S-1)
feasible1 = func.feasible(f_params, nvec_guess1, bvec_guess1)
print(feasible1)

print("2.(b) :")
nvec_guess2 = 0.95 * np.ones(S)
bvec_guess2 = np.append([0.0], np.ones(S-2))
feasible2 = func.feasible(f_params, nvec_guess2, bvec_guess2)
print(feasible2)

print("2.(c) :")
nvec_guess3 = 0.95 * np.ones(S)
bvec_guess3 = np.append([0.5], np.ones(S-2))
feasible3 = func.feasible(f_params, nvec_guess3, bvec_guess3)
print(feasible3)

print("2.(d) :")
nvec_guess4 = 0.5 * np.ones(S)
bvec_guess4 = \
  np.array([-0.01, 0.1, 0.2, 0.23, 0.25, 0.23, 0.2, 0.1,
     -0.01])
feasible4 = func.feasible(f_params, nvec_guess4, bvec_guess4)
print(feasible4)


# Q3

params = S, beta, sigma, A, alpha, delta, chi, b, mu, l_tilde, SS_tol
ss_output = func.get_SS(params, bvec_guess4, nvec_guess4, graphs)
print("n_ss : ", ss_output["n_ss"], "b_ss ： ", ss_output["b_ss"],
	"K_ss : ", ss_output["K_ss"], "C_ss", ss_output["C_ss"], "L_ss : ", ss_output["L_ss"])




# Q4
print ("Q4 :")
maxiter_TPI = 10000
mindist_TPI = 1e-9
xi = 0.2
TPI_tol = 1e-9
T = 25
b_ss = ss_output["b_ss"]
n_ss = ss_output["n_ss"]
K_ss = ss_output["K_ss"]
C_ss = ss_output["C_ss"]
L_ss = ss_output["L_ss"]
params = (S, T, beta, sigma, chi, b, mu, l_tilde, A, alpha, delta, b_ss, n_ss, K_ss, C_ss,
         L_ss, maxiter_TPI, mindist_TPI, xi, SS_tol)

print (get_TPI(params, bvec_guess4, True))





























