import numpy as np
from scipy.integrate import odeint
from VarMLFM_adapGrad import VarMLFM_adapGrad, VarMLFM_adapGrad_missing_data
import matplotlib.pyplot as plt

def g1(t):
    return np.cos(t)

A0 = np.array([[0, 0.],
               [0., 0]])
A1 = np.array([[0., -1.],
               [1., 0.]])

tt = np.linspace(0., 3., 17)
x0 = np.array([1., 0.])
sol = odeint(lambda x, t: np.dot(A0 + g1(t)*A1, x), x0, tt)
N = 3
redInd = np.linspace(0, tt.size-1, N, dtype=np.int)

obj1 = VarMLFM_adapGrad([A0, A1],
                        tt[redInd], sol[redInd, ],
                        [0.05, 0.05],
                        [0.1, 0.1],
                        [[1., 1.], [1., 1.]])

obj2 = VarMLFM_adapGrad_missing_data([A0, A1],
                                     tt, redInd, sol[redInd, ],
                                     [0.05, 0.05],
                                     [0.1, 0.1],
                                     [[1., 1.], [1., 1.]])


fig = plt.figure()
ax = fig.add_subplot(111)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

for nt in range(25):
    obj1._update_X_var_dist()
    obj2._update_X_var_dist()
    obj1._update_G_var_dist()
    obj2._update_G_var_dist()

    ax.plot(obj1.data_times, obj1._X_means[0], 'b-', alpha=0.2)
    ax.plot(obj2.full_times, obj2._X_means[0], 'r-', alpha=0.2)

    ax2.plot(obj1.data_times, obj1._G_means[0], 'b+', alpha=0.2)
    ax2.plot(obj2.full_times, obj2._G_means[0], 'r-+', alpha=0.2)

ax.plot(tt, sol[:, 0], 'k-')

ax.plot(tt[redInd], sol[redInd, 0], 's')

ax2.plot(tt, g1(tt), 'k-')
plt.show()
