import numpy as np
from gpode.latentforcemodels import VarMLFM
from scipy.integrate import quad, dblquad, odeint
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
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

vobj = VarMLFM(5, tt, yy, .15,
               As=[A0, A1, A2],
               lscales=[1.5, 1.5],
               obs_noise_priors=[[1., .5],
               [1, .5]])



tt = []
X = []
for i in range(16):
    t, v = vobj._func(i)
    tt.append(t)
    X.append(v)

X = np.array(X)

plt.plot(tt, np.cos(tt), 'o')
plt.plot(tt, np.sin(tt), 's')
plt.plot(tt, X, 'k+')

plt.show()
