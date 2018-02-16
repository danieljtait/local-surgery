import numpy as np
from gpode import latentforcemodels as lfm
from gpode.examples import DataLoader
import matplotlib.pyplot as plt
from gpode.bayes import Parameter, ParameterCollection
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

np.set_printoptions(precision=4)

MLFM = lfm.MulLatentForceModel_adapgrad

tt = np.linspace(.0, 5., 12)
mathieudat = DataLoader.load("mathieu", 57, tt, [0.1, 0.1], a=1., h=.9)
np.random.seed(3)

K = 2
xkp = []
for k in range(K):
    p1 = Parameter("phi_{}_1".format(k),
                   prior=("gamma", (4, 0.2)),
                   proposal=("normal rw", 0.1))
    p2 = Parameter("phi_{}_2".format(k),
                   prior=("gamma", (4, 0.2)),
                   proposal=("normal rw", 0.1))
    phi_k = ParameterCollection([p1, p2], independent=True)
    xkp.append(phi_k)


gkp = []
for r in range(1):
    p1 = Parameter("psi_{}_1".format(r+1),
                   prior=("gamma", (4, 4.)),
                   proposal=("normal rw", 0.1))
    p2 = Parameter("psi_{}_2".format(r+1),
                   prior=("gamma", (4, 4.)),
                   proposal=("normal rw", 0.1))
    psi_r = ParameterCollection([p1, p2], independent=True)
    gkp.append(psi_r)


# Make the obs. noise parameters
sigmas = [Parameter("sigma_{}".format(k),
                    prior=("gamma", (1, .05)),
                    proposal=("normal rw", 0.1))
          for k in range(K)]

# Make the grad. noise parameters
gammas = [Parameter("gamma_{}".format(k),
                    prior=("gamma", (1, 0.05)),
                    proposal=("normal rw", 0.1))
          for k in range(K)]

m = MLFM(xkp, gkp,
         sigmas, gammas,
         As=mathieudat["As"],
         data_time=mathieudat["time"],
         data_Y=mathieudat["Y"])

m._Gs = [np.ones(mathieudat["time"].size)] + mathieudat["Gs"]
X = m.data.Y.copy()
m._X = X

X = []
SIGMAS = []
G1s = []

fig = plt.figure()
ax = fig.add_subplot(111)

Z = []

for nt in range(3000):
    m.gibbs_update_step()
    if nt % 20 == 0 and nt > 1000:
        ax.plot(m.data.time, m._X[:, 0], 'k+')
        X.append(m._X[:, 0].copy())
        print("====", nt, "====")
        for r in [0]:
            print(m._g_kernels[r].kpar.value())
        for k in range(2):
            print(m._x_kernels[k].kpar.value())
        G1s.append(m._Gs[1].copy())
        for s in m.sigmas:
            print(s.value)
        print(np.array([_g.value for _g in m.gammas]))

        Z.append(m._dXdt(m._X, m._Gs)[:, 0] - m.data.Y[:, 1])

    SIGMAS.append([s.value for s in m.sigmas])
ax.plot(m.data.time, m.data.Y[:, 0], 's')


fig = plt.figure()
ax = fig.add_subplot(111)

G1s = np.array(G1s)
par = m._g_kernels[0].kpar.value()


def objfunc(gg):
    _Gs = [_g.copy() for _g in m._Gs]
    _Gs[1] = gg
    lval = m._log_eq20(m._X, _Gs)

    lprior = multivariate_normal.logpdf(gg, mean=np.zeros(gg.size),
                                        cov=m._g_kernels[0].cov(tt,
                                                                tt,
                                                                kpar=par))
    return -(lval + lprior)


res = minimize(objfunc, m._Gs[1])
var = np.diag(res.hess_inv)
sd = np.sqrt(var)

ax.fill_between(m.data.time,
                res.x + 2*sd,
                res.x - 2*sd, alpha=0.2)
ax.plot(m.data.time, res.x, 'gs-.')
ax.plot(m.data.time, np.mean(G1s, axis=0))
ax.plot(mathieudat["time"], mathieudat["Gs"][0], 's')

for i, g in enumerate(G1s):
    if i % 2 == 0:
        ax.plot(m.data.time, g, 'k+', alpha=0.2)
ax.set_ylim((-5., 5.))

plt.show()
