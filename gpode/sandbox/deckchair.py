import numpy as np
from gpode.bayes import Parameter
from gpode.kernels import SEParameterCollection, Kernel
from scipy.stats import multivariate_normal

np.set_printoptions(precision=4)
np.random.seed(11)

p1 = Parameter("psi_1",
               prior=("gamma", (4, 0.2)),
               proposal=("normal rw", 0.1))

p2 = Parameter("psi_2",
               prior=("gamma", (4, 0.2)),
               proposal=("normal rw", 0.1))

psi = SEParameterCollection(p1, p2, independent=True)

kern = Kernel.SquareExponKernel(psi)
p1.value = 0.24
p2.value = 1.11

tt = np.linspace(0., 2., 5)
C = kern.cov(tt)
L = np.linalg.cholesky(C)
z = np.dot(L, np.random.normal(size=tt.size))


def ell(psi_val):
    cov = kern.cov(tt, kpar=psi_val)
    return multivariate_normal.logpdf(z, mean=np.zeros(tt.size),
                                      cov=cov)


def dLdC(y, C):
    Cinv = np.linalg.inv(C)
    return -0.5*(Cinv - np.dot(Cinv, np.dot(np.outer(y, y), Cinv)))


def dCdpsi(psi, psi0):
    S, T = np.meshgrid(tt, tt)
    res = -psi0*(T-S)**2*np.exp(-psi*(S-T)**2)
    return res.reshape(S.shape)


l = ell(psi.value())

eps = 1e-6
psip = psi.value()
psip[0] += eps

print((ell(psip)-l)/eps)
print(np.sum(dLdC(z, C) * C/psi.value()[0]))

psip = psi.value()
psip[1] += eps

print((ell(psip)-l)/eps)
print(np.sum(dLdC(z, C) * dCdpsi(psi.value()[1], psi.value()[0])))
