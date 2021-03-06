import numpy as np
from gpode import latentforcemodels as lfm
from gpode.examples import DataLoader
import matplotlib.pyplot as plt
from gpode.bayes import Parameter, ParameterCollection
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d


def _norm_quad_form(x, L):
    return -0.5*np.dot(x, np.linalg.solve(L.T, np.linalg.solve(L, x)))


def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))


np.set_printoptions(precision=4)

MLFM = lfm.MulLatentForceModel_adapgrad

tt = np.linspace(.0, 5., 10)
mathieudat = DataLoader.load("mathieu", 57, tt, [0.1, 0.1], a=1., h=.9)
np.random.seed(3)

K = 2
xkp = []
for k in range(K):
    p1 = Parameter("phi_{}_1".format(k),
                   prior=("gamma", (4, 0.2)),
                   proposal=("normal rw", 0.1))
    p2 = Parameter("phi_{}_2".format(k),
                   prior=("gamma", (4, 0.2)),
                   proposal=("normal rw", 0.1))
    phi_k = ParameterCollection([p1, p2], independent=True)
    xkp.append(phi_k)


gkp = []
for r in range(1):
    p1 = Parameter("psi_{}_1".format(r+1),
                   prior=("gamma", (4, 4.)),
                   proposal=("normal rw", 0.1))
    p2 = Parameter("psi_{}_2".format(r+1),
                   prior=("gamma", (4, 4.)),
                   proposal=("normal rw", 0.1))
    psi_r = ParameterCollection([p1, p2], independent=True)
    gkp.append(psi_r)


# Make the obs. noise parameters
sigmas = [Parameter("sigma_{}".format(k),
                    prior=("gamma", (1, .05)),
                    proposal=("normal rw", 0.1))
          for k in range(K)]

# Make the grad. noise parameters
gammas = [Parameter("gamma_{}".format(k),
                    prior=("gamma", (1, 0.05)),
                    proposal=("normal rw", 0.1))
          for k in range(K)]

m = MLFM(xkp, gkp,
         sigmas, gammas,
         As=mathieudat["As"],
         data_time=mathieudat["time"],
         data_Y=mathieudat["Y"])

for s in sigmas:
    s.value = 0.1
for g in gammas:
    g.value = 0.00001

m._store_gpdx_covs()


m._Gs = [np.ones(mathieudat["time"].size)] + mathieudat["Gs"]
X = mathieudat["X"]
m._X = X

Lxx1 = m.Lxx[0]
Lxx2 = m.Lxx[1]
Cxdx1 = m.Cxdx[0]
Cxdx2 = m.Cxdx[1]
S_chol1 = m.S_chol[0]
S_chol2 = m.S_chol[1]
g = m._Gs[1]
G = np.diag(g)
B1 = np.dot(Cxdx1.T, _back_sub(Lxx1, np.diag(np.ones(tt.size))))
B2 = np.dot(Cxdx2.T, _back_sub(Lxx2, np.diag(np.ones(tt.size))))

I = np.diag(np.ones(tt.size))
S1 = np.dot(S_chol1, S_chol1.T)
S2 = np.dot(S_chol2, S_chol2.T)
S1inv = np.linalg.inv(S1)
S2inv = np.linalg.inv(S2)

C11 = np.dot(B1.T, np.dot(S1inv, B1)) + np.dot(G, np.dot(S2inv, G))
C12 = -np.dot(B1.T, S1inv) - np.dot(G, np.dot(S2inv, B2))
C22 = S1inv + np.dot(B2.T, np.dot(S2inv, B2))
Cinv = np.row_stack((np.column_stack((C11, C12)),
                     np.column_stack((C12.T, C22))))
C = np.linalg.inv(Cinv)
mat1 = np.row_stack((np.column_stack((-B1, I)),
                     np.column_stack((G, -B2))))
mat2 = np.row_stack((np.column_stack((S1inv, np.zeros(I.shape))),
                     np.column_stack((np.zeros(I.shape), S2inv))))
#print(np.dot(mat1.T, np.dot(mat2, mat1)))
#print(Cinv)
#print(C)


def logq(x1, x2):

    m1 = np.dot(B1, x1)  # np.dot(Cxdx1.T, _back_sub(Lxx1, x1))
    m2 = np.dot(B2, x2)

    eta1 = x2 - m1
    eta2 = g*x1 - m2

    exp_arg = _norm_quad_form(eta1, S_chol1) + _norm_quad_form(eta2, S_chol2)
    return exp_arg


x1 = np.random.normal(size=tt.size)
x2 = np.random.normal(size=tt.size)
xx = np.concatenate((x1, x2))
print(logq(x1, x2))
print(-0.5*np.dot(xx, np.dot(Cinv, xx)))
#print(_norm_quad_form(xx, np.linalg.cholesky(C)))

print("====================")
A0 = mathieudat["As"][0]
A1 = mathieudat["As"][1]

gobs = mathieudat["Gs"][0]


def dXdt(X, t, g):
    return np.dot(A0 + g(t)*A1, X)


X = []
for nt in range(10):

    
    
    sol = odeint(dXdt, tt, m._X[0, ], args=(pred, ))


fig = plt.figure()
ax = fig.add_subplot(111)

plt.show()


print("====================")
"""
N = m._X.shape[0]
x0 = np.concatenate((m._X[:, 0], m._X[:, 1]))

res_nm = minimize(lambda z: -logq(z[:N], z[N:]), x0, method="Nelder-Mead")
res = minimize(lambda z: -logq(z[:N], z[N:]), res_nm.x)

print(res_nm.x)
print(res.x[:N])
print(res_nm.fun)
print(res.fun)
print(-logq(np.zeros(N), np.zeros(N)))
"""
