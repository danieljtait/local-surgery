import numpy as np
from gpode.latentforcemodels import NestedIntegralKernel
from gpode.examples import DataLoader
from scipy.integrate import quad
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt


np.set_printoptions(precision=4)

Jkernel = NestedIntegralKernel()

tt = np.linspace(.0, 1.5, 12)
mathieudat = DataLoader.load("mathieu", 57, tt, [0.1, 0.1], a=1., h=.9)
np.random.seed(4)


def integrate(dt, J1J2, x0):

    A = mathieudat["As"][0]
    B = mathieudat["As"][1]
    AA = np.dot(A, A)
    AB = np.dot(A, B)
    BA = np.dot(B, A)
    BB = np.dot(B, B)

    X = []
    for t, J in zip(dt, J1J2):
        S1 = np.dot(A*t + B*J[0], x0)
        S2 = np.dot(AB*J[1] + BA*(t*J[0] - J[1]), x0)
        S3 = 0.5*np.dot(AA*t**2 + BB*J[0]**2, x0)

        X.append(x0 + S1 + S2 + S3)

    return np.array(X)


def objfunc(j1j2, dt, x0, cov):
    J1J2 = j1j2.reshape((tt.size, 2))
    ngm_sol = integrate(dt, J1J2, x0)

    Y = mathieudat["Y"]

    val1 = sum(norm.logpdf(Y[:, 0], ngm_sol[:, 0], scale=0.1))
    val2 = sum(norm.logpdf(Y[:, 1], ngm_sol[:, 1], scale=0.1))

    _j1_j2 = np.concatenate((J1J2[:, 0], J1J2[:, 1]))
    prior_val = multivariate_normal.logpdf(_j1_j2,
                                           mean=np.zeros(j1j2.size),
                                           cov=cov)

    return -(val1 + val2 + prior_val)


i0 = 0
t0 = tt[i0]
x0 = mathieudat["X"][i0, ]

dt = tt - t0
c11 = Jkernel.cov(1, 1, dt, dt)
c12 = Jkernel.cov(1, 2, dt, dt)
c22 = Jkernel.cov(2, 2, dt, dt)
C = np.row_stack((
    np.column_stack((c11, c12)),
    np.column_stack((c12.T, c22))))
C += np.diag(1e-5*np.ones(C.shape[0]))


J1 = np.array([quad(lambda t: 2*0.9**2*np.cos(2*t), 0, _dt)[0] for _dt in dt])
J2 = np.array([quad(lambda t: quad(lambda s: 2*0.9**2*np.cos(2*s),
                                   0, t)[0], 0, _dt)[0] for _dt in dt])
J1J2 = np.column_stack((J1, J2))

ngm_sol = integrate(dt, J1J2, x0)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(mathieudat["time"], mathieudat["X"])
ax.plot(dt + t0, ngm_sol, 'o')

res = minimize(objfunc, J1J2.ravel(), args=(dt, x0, C))
res_J1J2 = res.x.reshape((tt.size, 2))
res_sol = integrate(dt, res.x.reshape(tt.size, 2), x0)

ax.plot(dt + t0, res_sol)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(dt + t0, res_J1J2[:, 0])
ax.plot(dt + t0, J1, 'o')

plt.show()
