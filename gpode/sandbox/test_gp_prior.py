import numpy as np
from gpode.latentforcemodels import VariationalMLFM2
from scipy.integrate import quad, dblquad, odeint
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.set_printoptions(precision=2)

A0 = np.array([[0., 0.], [0., 0.]])
A1 = np.array([[0., -1.], [1., 0.]])
A2 = np.array([[ 0., 0.], [0., 1.]])

g1 = lambda t: np.cos(t)
g2 = lambda t: np.exp(-0.5*(t-2)**2)

x0 = np.array([1., 0.0])
ttDense = np.linspace(0., 3., 50)

sol = odeint(lambda x, t: np.dot(A0 + A1*g1(t) + g2(t)*A2, x), x0, ttDense)
redInd = np.linspace(0, ttDense.size-1, 9, dtype=np.intp)
tt = ttDense[redInd]
yy = sol[redInd, ]

vobj = VariationalMLFM2(5, tt, yy, .1,
                        As=[A0, A1, A2],
                        lscales=[1.0, 5.5],
                        obs_noise_priors=[[2000, .5],
                                          [2500, .5]])

def get_cov_pq(p, q, ta, tb, r):
    key = "J{}J{}".format(p, q)

    Ta, Sa = np.meshgrid(ta, ta)
    Tb, Sb = np.meshgrid(tb, tb)
    
    return vobj._cov_funcs[key](Sb.ravel(), Tb.ravel(), Sa.ravel(), Ta.ravel(), 1., vobj.lscales[r]).reshape(Ta.shape)


fig = plt.figure()
ax = fig.add_subplot(111)
for nt in range(5):
    ta = np.concatenate((vobj.backward_full_ts[:-1], vobj.forward_full_ts[:-1]))
    tb = np.concatenate((vobj.backward_full_ts[1:], vobj.forward_full_ts[1:]))

    Cgg = get_cov_pq(0, 0, ta, tb, 0)
    CgJ = get_cov_pq(0, 1, ta, tb, 0)
    CJJ = get_cov_pq(1, 1, ta, tb, 0)
    print(np.diag(np.linalg.inv(CJJ)))
    C = np.row_stack((np.column_stack((Cgg, CgJ)),
                      np.column_stack((CgJ.T, CJJ))))
    try:
        L = np.linalg.cholesky(C)
    except:
        C += np.diag(1e-6*np.ones(C.shape[0]))
        L = np.linalg.cholesky(C)
    z = np.dot(L, np.random.normal(size=C.shape[0]))
    ax.plot(tb, z[:Cgg.shape[0]], 'k+', alpha=0.2)

print(np.diag(CJJ))

plt.show()
    
