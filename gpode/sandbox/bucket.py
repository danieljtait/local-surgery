import numpy as np
from gpode.examples import DataLoader
import matplotlib.pyplot as plt
from scipy.special import jn
from scipy.misc import derivative as deriv


order = 2
tt = np.linspace(0.5, 10., 10)
bd = DataLoader.load("bessel jn", 11, tt, [0.1, 0.05],
                     order=order)


fig = plt.figure()
ax = fig.add_subplot(111)
td = np.linspace(tt[0], tt[-1], 100)
ax.plot(td, jn(order, td), 'k-', alpha=0.2)
ax.plot(td, deriv(lambda z: jn(order, z), x0=td, dx=1e-6),
        'k-.', alpha=0.2)
ax.plot(bd["time"], bd["Y"], 's')

plt.show()
