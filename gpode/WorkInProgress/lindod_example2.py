import numpy as np
import gpode.kernels
from gpode.kernels import Kernel, KernelParameter
import gpode.gaussianprocesses
from scipy.integrate import quad, odeint
from scipy.stats import gamma, norm, multivariate_normal
from lindod_src import (MGPLinearAdapGrad2, Data, state_update,
                        log_updf, gr_post_conditional,
                        xk_post_conditional, lgp_hyperpar_pdf,
                        _x_gp_pars)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from testfunctions import *


A0 = np.array([[0.0, 0.],
               [0.0, 0.]])
A1 = np.array([[0.0, -1.],
               [1.0, 0.]])

x0 = np.array([1., 0.])
tt = np.linspace(0., 3., 5)
Y = odeint(lambda x, t: np.dot(A0 + np.cos(t)*A1, x), x0, tt)

data = Data(tt, Y)
sigmas = [0.1, 0.1]

# Initalise the kernel hyperparameters
kpars = []
for r in range(Y.shape[1]):
    kp = KernelParameter(lambda x: [1., x], "phi"+str(r+1),
                         prior=("gamma", [4., 0.2]),
                         proposal=("normal rw", 0.1))
    kpars.append(kp)

# Initalise the gamma parameters

mod = MGPLinearAdapGrad2([A0, A1], data, sigmas, kpars)
mod.model_setup(Gs=[np.cos(tt)])
mod.gammas = [0.1, 0.1]

kse = Kernel.SquareExponKernel()
mod.Gs_kernels = [kse]
mod._psi_r_mh_update(0)

for k, kern in enumerate(mod.latent_gp_kernels):
    mod._update_stored_xgp_pars(k, kern)


test2(mod)

"""
fig = plt.figure()
ax = fig.add_subplot(111)
sim_kpar = []
N_SIM = 1000
for nt in range(N_SIM):
    for k in range(2):
        mod._x_k_update(k)
    for k in range(2):
        mod._phi_k_mh_update(k)
#    for r in [0]:
#        mod._psi_r_mh_update(r)
#    for r in [0]:
#        mod._g_r_update(r)
    if nt > N_SIM/2 and nt % 10 == 0:
        sim_kpar.append(mod.latent_gp_kernels[1].kpar.value)
        ax.plot(tt, mod.latent_X[:, 1], 'k+', alpha=0.2)
        #ax.plot(tt, mod.Gs[1], 'k+', alpha=0.2)
    if nt % 100 == 0:
        print(nt)

ax.plot(tt, Y[:, 1], 's')
"""
"""
sim_kpar = np.array(sim_kpar)
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(sim_kpar)
"""
#plt.show()

"""
def objfunc(x):
    try:
        return -lgp_hyperpar_pdf(x, 0, prior,
                                 kernels[0], tt,
                                 mod.As, mod.Gs,
                                 mod.sigmas, mod.gammas,
                                 mod.latent_X, mod._dXdt)
    except:
        return np.inf
res = minimize(objfunc, 0.5)
print(res)
"""
