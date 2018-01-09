import numpy as np
from scipy.special import erf
from .kernels import CollectionKernel


ROOTPI = np.sqrt(np.pi)


def hqp(s, t, Dp, Dq, l, nu):
    expr1 = np.exp(Dq*t)*(erf((s-t)/l) + erf(t/l+nu))
    expr2 = np.exp(-Dp*t)*(erf(s/l - nu) + erf(nu))
    return np.exp(nu**2 - Dq*s)*(expr1 - expr2)/(Dp + Dq)


def kfunc(s, t, ind1=0, ind2=0, **kwargs):
    S = kwargs['S']
    D = kwargs['D']
    lScales = kwargs['lScales']
    p = ind1
    q = ind2
    res = np.array([0.5*ROOTPI*S[r,p]*S[r,q]*l*(hqp(s, t, D[p], D[q], l, 0.5*l*D[q])
                                                + hqp(t, s, D[q], D[p], l, 0.5*l*D[p]))
                    for r, l in enumerate(lScales)])
    
    return np.sum(res, axis=0)


def llfmSquareExponKernel(size):
    S = np.diag(np.ones(size))
    D = np.ones(size)
    lScales = np.ones(size)
    return CollectionKernel(kfunc, size, {'lScales': lScales, 'S': S, 'D': D})



