import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# Problem 3


def Prob3_Euler(n, *args):
    w, r, b_S, sigma, chi, theta = args
    c = (1 + r) * b_S + w * n
    error = w * (c ** (-sigma)) - chi * (n ** (1 / theta))

    return error


w = 1.0
r = 0.1
b_S = 1.0
sigma = 2.2
chi = 1.0
theta = 2.0
Prob3_args = (w, r, b_S, sigma, chi, theta)
n_guess = 0.5

n_S = opt.fsolve(Prob3_Euler, n_guess, args=(Prob3_args))
print('Prob 3: n_S = ', n_S)
c = (1 + r) * b_S + w * n_S
error = w * (c ** (-sigma)) - chi * (n_S ** (1 / theta))
print('Prob 3: error = ', error)


# Problem 4


def MU_sumsq(CRRA_params, *args):
    chi_CRRA, sigma = CRRA_params
    theta, chi_CFE, labor_sup = args

    MU_CFE = chi_CFE * (labor_sup ** (1 / theta))
    MU_CRRA = chi_CRRA * ((1 - labor_sup) ** (-sigma))
    sumsq = ((MU_CRRA - MU_CFE) ** 2).sum()

    return sumsq


theta = 2.0
chi_CFE = 1.0
labor_min = 0.1
labor_max = 0.9
labor_N = 1000
labor_sup = np.linspace(labor_min, labor_max, labor_N)
fit_args = (theta, chi_CFE, labor_sup)
chi_CRRA_init = 0.5
sigma_init = 2.8
CRRA_init = np.array([chi_CRRA_init, sigma_init])
bnds_CRRA = ((1e-12, None), (1e-12, None))
CRRA_params = opt.minimize(MU_sumsq, CRRA_init, args=(fit_args),
                           method='L-BFGS-B', bounds=bnds_CRRA)
chi_CRRA, sigma = CRRA_params.x
sumsq = CRRA_params.fun
if CRRA_params.success:
    print('SUCCESSFULLY ESTIMATED CRRA DISUTILITY OF LABOR FUNCTION.')
    print('chi_CRRA=', chi_CRRA, ' sigma=', sigma, ' SumSq=', sumsq)


# Problem 5

MU_CRRA = chi_CRRA * ((1 - labor_sup) ** (-sigma))
MU_CFE = chi_CFE * (labor_sup ** (1 / theta))
fig, ax = plt.subplots()
plt.plot(labor_sup, MU_CRRA, label='CRRA MU')
plt.plot(labor_sup, MU_CFE, label='CFE MU')
# for the minor ticks, use no labels; default NullFormatter
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('CFE marginal utility versus fitted CRRA',
          fontsize=20)
plt.xlabel(r'Labor supply $n_{s,t}$')
plt.ylabel(r'Marginal disutility')
plt.xlim((0, 1))
# plt.ylim((-1.0, 1.15 * (b_ss.max())))
plt.legend(loc='upper left')
plt.savefig('CRRAvsCFE_MU')
plt.show()
plt.close()
