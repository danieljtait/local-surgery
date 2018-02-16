import numpy as np
from gpode.latentforcemodels import (NestedIntegralKernel,
                                     NeumannGenerativeModel,
                                     VariationalMLFM)
from gpode.examples import DataLoader
from scipy.integrate import quad
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from scipy.special import mathieu_cem


np.set_printoptions(precision=2)

Jkernel = NestedIntegralKernel(origin="recursive")

tt = np.linspace(.0, 1.5, 5)
mathieudat = DataLoader.load("mathieu", 40, tt, [0.0001, 0.0001], a=1., h=1.)
np.random.seed(4)

Y = mathieudat["Y"]

xkp = [np.array([1., 5.5]),
       np.array([1., 5.5])]

gkp = [np.array([1., 3.])]

vobj = VariationalMLFM(g_kernel_pars=gkp,
                       x_kernel_pars=xkp,
                       sigmas=np.array([0.01, 0.01]),
                       gammas=np.array([0.1, 0.1]),
                       As=mathieudat["As"],
                       data_time=tt,
                       data_Y=Y)
vobj._store_gpdx_covs()


C00 = vobj._x_kernels[0].cov(0, 0, tt)
C01 = vobj._x_kernels[0].cov(0, 1, tt)
C11 = vobj._x_kernels[0].cov(1, 1, tt)

Sxx = np.row_stack((np.column_stack((C00, C01)),
                    np.column_stack((C01.T, C11))))
mxx = np.ravel(Y.T)

Sgg = C00.copy()
mgg = np.random.normal(size=tt.size)

mg, cg_i = vobj._parse_component_k_for_g(1, mxx, Sxx)

mx, cx_i = vobj._parse_component_k_for_x(1, mg, np.linalg.inv(cg_i))
#
mg, Sg = vobj._get_g_conditional(mx, np.linalg.inv(cx_i))
mx, Sx = vobj._get_x_conditional(mg, Sg)
#vobj.func(1, mg, Sg)


fig0 = plt.figure()
ax = fig0.add_subplot(111)
tt_dense = np.linspace(tt[0], tt[-1], 100)
ax.plot(tt_dense, 2*np.cos(2*tt_dense), 'k-')
for nt in range(5):
    print(nt)
    mg, Sg = vobj._get_g_conditional(mx, Sx)
    mx, Sx = vobj._get_x_conditional(mg, Sg)
    ax.plot(tt, mg, 'k-+', alpha=(0.1*(nt+1)/15))
    print(mx.reshape(2, tt.size)[0, ])
#ax.plot(tt, mathieudat["Gs"][0], 's-')
#mobj = NeumannGenerativeModel(mathieudat["As"], tt, Y)
#mobj._set_times()
print(Sg)
sd = np.sqrt(np.diag(Sg))
ax.fill_between(tt, mg + 2*sd, mg-2*sd, alpha=0.2)



#print(mathieu_cem(1, 1., 0.))
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.plot(tt, mathieu_cem(1, 1., tt), '-.')
ax.plot(tt, mathieudat["Y"], 's')
ax.plot(tt, mx.reshape(2, tt.size)[0, ], '+-')
ax.plot(tt, mx.reshape(2, tt.size)[1, ], '+-')
ax.plot(tt, mathieudat["X"], 'ko')
plt.show()

"""
def integrate_recursive_forward(tb_vec, ta_vec, J1J2, x0):
    assert(all(tb_vec >= ta_vec))
    A = mathieudat["As"][0]
    B = mathieudat["As"][1]
    AA = np.dot(A, A)
    AB = np.dot(A, B)
    BA = np.dot(B, A)
    BB = np.dot(B, B)

    X = [x0]
    for nt, (tb, ta) in enumerate(zip(tb_vec, ta_vec)):

        j1 = J1J2[nt, 0]
        j2 = J1J2[nt, 1]

        x1new = np.dot(A*(tb-ta) + B*J1J2[nt, 0], X[-1])
        x2new = np.dot(AB*j2 + BA*((tb-ta)*j1 - j2), X[-1])
        x3new = 0.5*np.dot(AA*(tb-ta)**2 + BB*J1J2[nt, 0]**2, X[-1])

        X.append(X[-1] + x1new + x2new + x3new)

    return np.array(X)


tta = tt[:-1]
ttb = tt[1:]

J1J2 = []
for tb, ta in zip(ttb, tta):
    j1 = quad(lambda t: 2*0.9**2*np.cos(2*t), ta, tb)[0]
    j2 = quad(lambda t: quad(lambda s: 2*0.9**2*np.cos(2*s), ta, t)[0],
              ta, tb)[0]
    J1J2.append([j1, j2])
J1J2 = np.array(J1J2)

sol = integrate_recursive_forward(ttb, tta, J1J2, mathieudat["X"][0, ])

tpred = np.linspace(tt[0], tt[-1], 25)
C01 = Jkernel.cov(0, 1, tpred, np.zeros(tpred.size),
                  tb=ttb, ta=tta)
C02 = Jkernel.cov(0, 2, tpred, np.zeros(tpred.size),
                  tb=ttb, ta=tta)

C11 = Jkernel.cov(1, 1, ttb, tta)
C12 = Jkernel.cov(1, 2, ttb, tta)
C22 = Jkernel.cov(2, 2, ttb, tta)

Cab = np.column_stack((C01, C02))
Cbb = np.row_stack((np.column_stack((C11, C12)),
                    np.column_stack((C12.T, C22))))

a = np.concatenate((J1J2[:, 0], J1J2[:, 1]))
assert(all(a == J1J2.T.ravel()))

L = np.linalg.cholesky(Cbb)
print(L.shape)
pred = np.linalg.solve(L.T, np.linalg.solve(L, a))
pred = np.dot(Cab, pred)

plt.plot(tpred, pred, '+')
plt.plot(tpred, 2*0.9**2*np.cos(2*tpred), 'k-', alpha=0.2)

#plt.plot(tt, sol)
#plt.plot(tt, mathieudat["X"], 's')
plt.show()
"""
"""
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
"""
