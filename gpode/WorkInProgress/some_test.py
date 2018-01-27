import numpy as np
from scipy.stats import multivariate_normal


tt = np.linspace(0., 1., 3)
x = np.random.normal(size=3)


def C(s):
    T_, S_ = np.meshgrid(tt, tt)
    return np.exp(-s*(T_.ravel()-S_.ravel())**2).reshape(T_.shape)


def dCds(s):
    T_, S_ = np.meshgrid(tt, tt)
    res = -(T_.ravel()-S_.ravel())**2*np.exp(-s*(T_.ravel()-S_.ravel())**2)
    return res.reshape(T_.shape)


def ell(s):
    return multivariate_normal.logpdf(x, mean=np.zeros(tt.size), cov=C(s))


def dLdC(C):
    Cinv = np.linalg.inv(C)
    return -0.5*(Cinv - np.dot(Cinv, np.dot(np.outer(x, x,), Cinv)))


eps = 1e-6
s = 1.1
sp = s + eps

print((ell(sp)-ell(s))/eps)
print(np.sum(dLdC(C(s)) * dCds(s)))
