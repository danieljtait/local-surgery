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


def _integrate(psi, x0):
    X = [x0]
    I = np.diag(np.ones(x0.size))
    for f in psi:
        X.append(np.dot(I + A1*f[0] + A2*f[1], X[-1]))
    return np.array(X)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ttDense, sol, 'k-')

t0ind = 5
t0 = tt[t0ind]
x0 = yy[t0ind, ]

ss = np.linspace(t0, tt[-1], 15)
sa = ss[:-1]
sb = ss[1:]
Nf = sb.size

psi = []
for _sa, _sb in zip(sa, sb):
    J1 = quad(g1, _sa, _sb)[0]
    J2 = quad(g2, _sa, _sb)[0]
    psi.append([J1, J2])
psi = np.array(psi)
X = _integrate(psi, x0)



print(psi)
f = psi.ravel()
print(f.reshape(Nf, 2))

ax.plot(ss, X, '+')
ax.plot(tt[t0ind:], yy[t0ind:, :], 's')

plt.show()
    
