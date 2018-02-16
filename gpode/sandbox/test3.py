import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.integrate import quad, odeint
from gpode.latentforcemodels import (NestedIntegralKernel,
                                     NeumannGenerativeModel,
                                     VariationalMLFM)


def g1(t):
    return np.exp(-(t-2)**2)

def g2(t):
    return np.cos(t)

A0 = np.array([[0., 0.],
               [0.,-0.1]])
A1 = np.array([[1., 0.],
               [0., 0.]])
A2 = np.array([[0., 1.],
               [-1., 0.]])

x0 = np.array([1., 0.])
tt = np.linspace(0., 4.)

sol = odeint(lambda x, t: np.dot(A0 + A1*g1(t) + A2*g2(t), x),
             x0,
             tt)

nr = 9
redInd = np.linspace(0, tt.size-1, nr, dtype=np.intp)
rtt = tt[redInd]
Y = sol[redInd, ]


xkp = [np.array([1., 5.5]),
       np.array([1., 5.5])]

gkp = [np.array([1., 3.]),
       np.array([1., 3.])]

vobj = VariationalMLFM(g_kernel_pars=gkp,
                       x_kernel_pars=xkp,
                       sigmas=np.array([0.01, 0.01]),
                       gammas=np.array([0.1, 0.1]),
                       As = [A0, A1, A2],
                       data_time=rtt,
                       data_Y=Y)
vobj._store_gpdx_covs()


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, sol, 'k-', alpha=0.2)
plt.show()



