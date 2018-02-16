import numpy as np
from gpode.kernels import GradientMultioutputKernel
from scipy.integrate import odeint
from scipy.optimize import minimize
np.set_printoptions(precision=4)


def _norm_quad_form(x, L):
    return -0.5*np.dot(x, _back_sub(L, x))


def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))


def _g(t):
    return -np.cos(t)


def dXdt(X, t):
    return np.dot(np.array([[0., 1.],
                            [_g(t), 0.]]), X)


tt = np.linspace(0., 5., 15)
sol = odeint(dXdt, [1., 0.], tt)


gammas = [0.1, 0.1]
kern1 = GradientMultioutputKernel.SquareExponKernel([0.0157, 0.1232])
kern2 = GradientMultioutputKernel.SquareExponKernel([2.1737, 1.9016])

I = np.diag(np.ones(tt.size))
Cxx = kern1.cov(0, 0, tt)
Lxx = np.linalg.cholesky(Cxx)
Cxdx = kern1.cov(0, 1, tt)
Cdxdx = kern1.cov(1, 1, tt)
Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))
Sxx_inv = np.linalg.inv(Cdxdx_x + gammas[0]**2*I)
LSxx = np.linalg.cholesky(Cdxdx_x + gammas[0]**2*I)

Cyy = kern2.cov(0, 0, tt)
Lyy = np.linalg.cholesky(Cyy)
Cydy = kern2.cov(0, 1, tt)
Cdydy = kern2.cov(1, 1, tt)
Cdydy_y = Cdydy - np.dot(Cydy.T, _back_sub(Lyy, Cydy))
Syy_inv = np.linalg.inv(Cdydy_y + gammas[1]**2*I)
LSyy = np.linalg.cholesky(Cdydy_y + gammas[1]**2*I)

P1 = np.dot(Cxdx.T, _back_sub(Lxx, np.diag(np.ones(tt.size))))
P2 = np.dot(Cydy.T, _back_sub(Lyy, np.diag(np.ones(tt.size))))
I = np.diag(np.ones(tt.size))


def p(x, y, g):
    m1 = np.dot(Cxdx.T, np.dot(np.linalg.inv(Cxx), x))
    m2 = np.dot(Cydy.T, np.dot(np.linalg.inv(Cyy), y))

    f1 = y
    f2 = g*x

    expr1 = _norm_quad_form(f1-m1, LSxx) + _norm_quad_form(f2-m2, LSyy)

    return expr1


x = np.random.normal(size=tt.size)
y = np.random.normal(size=tt.size)
xy = np.concatenate((x, y))
g = _g(tt)

eta1 = y - np.dot(P1, x)
eta2 = g*x - np.dot(P2, y)

T1 = np.column_stack((-P1, I))
T2 = np.column_stack((np.diag(g), -P2))
T = np.row_stack((T1, T2))

K1aa = np.dot(P1.T, np.dot(Sxx_inv, P1))
K1ab = np.dot(-P1.T, Sxx_inv)
K1bb = Sxx_inv
K1 = np.row_stack((np.column_stack((K1aa, K1ab)),
                   np.column_stack((K1ab.T, K1bb))))

K2aa = np.dot(np.diag(g), np.dot(Syy_inv, np.diag(g)))
K2ab = np.dot(-np.diag(g), np.dot(Syy_inv, P2))
K2bb = np.dot(P2.T, np.dot(Syy_inv, P2))
K2 = np.row_stack((np.column_stack((K2aa, K2ab)),
                   np.column_stack((K2ab.T, K2bb))))


def func(pp, x, y, g, gammas):
    try:
        kp1 = pp[:2]
        kp2 = pp[2:]

        Cxx = kern1.cov(0, 0, tt, kpar=kp1)
        Lxx = np.linalg.cholesky(Cxx)
        Cxdx = kern1.cov(0, 1, tt, kpar=kp1)
        Cdxdx = kern1.cov(1, 1, tt, kpar=kp1)
        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))
        LSxx = np.linalg.cholesky(Cdxdx_x + gammas[0]**2*I)

        Cyy = kern2.cov(0, 0, tt, kpar=kp2)
        Lyy = np.linalg.cholesky(Cyy)
        Cydy = kern2.cov(0, 1, tt, kpar=kp2)
        Cdydy = kern2.cov(1, 1, tt, kpar=kp2)
        Cdydy_y = Cdydy - np.dot(Cydy.T, _back_sub(Lyy, Cydy))
        LSyy = np.linalg.cholesky(Cdydy_y + gammas[1]**2*I)

        m1 = np.dot(Cxdx.T, _back_sub(Lxx, x))
        m2 = np.dot(Cydy.T, _back_sub(Lyy, y))

        f1 = y
        f2 = g*x

        return -(_norm_quad_form(f1-m1, LSxx) + _norm_quad_form(f2-m2, LSyy))
    except:
        return np.inf


res = minimize(lambda z: func(z, sol[:, 0], sol[:, 1], _g(tt), [0.1, 0.1]),
               x0=np.ones(4),
               method="Nelder-Mead")

print(res)
"""

w = np.column_stack((-P1, I))
_K1 = np.dot(w.T, np.dot(Sxx_inv, w))

#print(K2bb)
Kinv = K1 + K2

#print(Kinv)
#K = np.linalg.inv(Kinv)
#ELL = np.linalg.cholesky(K)

print(p(x, y, g))
print(-0.5*np.dot(xy, np.dot(Kinv, xy)))
#print(np.dot(xy, np.dot(K2, xy)))
#print(_norm_quad_form(xy, ELL))
"""
