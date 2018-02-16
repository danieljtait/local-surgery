import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.random.seed(11)

A0 = np.array([[0., 1.],
               [-1., 0.]])

A1 = np.array([[0., 1.],
               [-1., 0.]])

A2 = np.array([[0., 0.],
               [0., 0.3]])

As = [A0, A1, A2]
"""
def S(phi):
    return sum([a*_f for a,_f in zip(As, phi)])

x0 = np.array([0.3, 0.1])
phi = [1., 0.3, -0.5]

S_ = S(phi)
print(np.dot(S_, x0))

#V = np.column_stack((np.dot(a, x0) for a in As))
#print(np.dot(V, phi))
"""


"""
def S(j1, j2):

    dt = 0.1
    
    result = np.diag(np.ones(2))
    result += A0*dt + A1*j1
#    result += 0.5*np.dot(A0, A0) + np.dot(A0, A1)*j2
#    result += np.dot(A1, A0)*(dt*j1 - j2) + 0.5*np.dot(A1, A1)*j1**2

    return result

x0 = np.array([1., 0.])

Jtrue = np.array([[0.5, 0.3],
                  [-.3, -0.4]])

S1 = S(*Jtrue[0, ])
S2 = S(*Jtrue[1, ])

y = np.dot(S2, np.dot(S1, x0)) + np.random.normal(size=2, scale=0.1)

def logp(j11, j12, j21, j22, y):
    S1 = S(j11, j12)
    S2 = S(j21, j22)
    mean = np.dot(S2, np.dot(S1, x0))
    return -0.5*sum((y-mean)**2)/(0.1**2)


def Elogp(j21, j22, Y):

    m1 = np.array([0., 0.])
    cov1 = np.array([[1., -0.3],
                     [-0.3, 0.6]])
    
#    def _integrand(x1, x2):
#        return logp(x1, x2, j21, j22, Y)*multivariate_normal.pdf([x1, x2], m1, cov1)

    def _integrand(x1):
        return logp(x1, 0, j21, 0., Y)*norm.pdf(x1,
                                                 loc=m1[0],
                                                 scale=np.sqrt(cov1[0, 0]))

    return quad(_integrand, -np.inf, np.inf)[0]

#    return dblquad(_integrand, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]


xs = np.linspace(-1.5, 1.5, 25)
ys = np.linspace(-0.5, 1.1, 5)

Z = np.zeros((xs.size, ys.size))

res = minimize(lambda z: -Elogp(z, -0.3, y), 0.)
print(res)
#for i, x1 in enumerate(xs):
#    for j, x2 in enumerate(ys):
#        Z[i, j] = Elogp(x1, x2, y)
#        print(i, j)
print(res.hess_inv)
plt.plot(xs, [Elogp(z, -0.3, y) for z in xs])
plt.plot(xs, -0.5*(xs-res.x)**2)
#plt.contour(xs, ys, Z)
plt.show()
"""
