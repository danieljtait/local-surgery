import numpy as np
from gpode.latentforcemodels import VariationalMLFM2
from scipy.integrate import quad
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)

tt = np.linspace(0., 2*np.pi, 6)
yy = tt.copy()


fig = plt.figure()
ax = fig.add_subplot(111)

for l in [0.1, 0.5, 1., 2., 5., 8., 10.]:
    try:
        vobj = VariationalMLFM2(2, tt, yy, 0.15,
                                As=[None, None, None],
                                lscales=[l, 1.])

        ss = np.linspace(0., 2*np.pi, 13)

        b_ta = vobj.backward_full_ts[:-1]
        b_tb = vobj.backward_full_ts[1:]
        f_ta = vobj.forward_full_ts[:-1]
        f_tb = vobj.forward_full_ts[1:]
        
        bJ = np.array([quad(lambda t: np.cos(t), _ta, _tb)[0]
                       for _ta, _tb in zip(b_ta, b_tb)])
        fJ = np.array([quad(lambda t: np.cos(t), _ta, _tb)[0]
                       for _ta, _tb in zip(f_ta, f_tb)])

        pred, var = vobj.pred_latent_force(0, ss, bJ, fJ, True)

        sd = np.sqrt(np.diag(var))
        
        ax.plot(ss, pred, '+')
        ax.fill_between(ss, pred + 2*sd, pred - 2*sd, alpha=0.2)
    except:
        print("{} failed".format(l))
ax.plot(vobj.backward_full_ts, np.cos(vobj.backward_full_ts), 'o')
ax.plot(vobj.forward_full_ts, np.cos(vobj.forward_full_ts), 'o')
ax.plot(ss, np.cos(ss), '-')
plt.show()
