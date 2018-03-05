import numpy as np
from scipy.linalg import block_diag
from scipy.integrate import odeint
from deleteme_src3 import (SimpleVarMLFM,
                           _get_Vi_cov,
                           _get_Vi_mean,
                           _parse_component_i_for_g,
                           _parse_component_i_for_x)
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

A0 = np.array([[0., 0.], [0., 0.]])
A1 = np.array([[0., -1.], [1., 0.]])
A2 = np.array([[ .0, 0.], [0., 1.]])
A = np.array([A0, A1, A2])

g1 = lambda t: np.cos(t)
g2 = lambda t: np.exp(-0.5*(t-2)**2)

def dXdt(X, t):
    return np.dot(A0 + A1*g1(t) + A2*g2(t), X)

tt = np.linspace(0., 3., 5)
x0 = np.array([0., 1.])
sol = odeint(dXdt, x0, tt)
Y = sol.copy()

obj = SimpleVarMLFM(A, tt, Y,
                    [0.01, 0.01], [.1, .1],
                    [[1., 1.], [1., 1.]])

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)

for nt in range(5):
    obj._update_X_var_dist()
    obj._update_G_var_dist()
    for m, c in zip(obj._X_means, ['b', 'r']):
        ax.plot(tt, m, c, alpha=0.2)
    for mg, c in zip(obj._G_means, ['b', 'r']):
        ax2.plot(tt, mg, c, alpha=0.2)

ax.plot(tt, sol, 's')
ax2.plot(tt, g1(tt), 'bs')
ax2.plot(tt, g2(tt), 'rs')
plt.show()
                         






