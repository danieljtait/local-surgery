import numpy as np
from gpode import latentforcemodels as lfm
from gpode.examples import DataLoader
import matplotlib.pyplot as plt
from gpode.bayes import Parameter, ParameterCollection


np.set_printoptions(precision=4)

MLFM = lfm.MulLatentForceModel_adapgrad

tt = np.linspace(.0, 3., 6)
mathieudat = DataLoader.load("mathieu", 5, tt, [0.05, 0.05], h=1.4)
np.random.seed(4)

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
                   prior=("gamma", (4, 0.2)),
                   proposal=("normal rw", 0.1))
    p2 = Parameter("psi_{}_2".format(r+1),
                   prior=("gamma", (4, 0.2)),
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
for nt in range(2000):
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
    SIGMAS.append([s.value for s in m.sigmas])
ax.plot(m.data.time, m.data.Y[:, 0], 's')


fig = plt.figure()
ax = fig.add_subplot(111)

G1s = np.array(G1s)
for g in G1s:
    ax.plot(m.data.time, g, 'k+')

ax.plot(m.data.time, np.mean(G1s, axis=0))
ax.plot(mathieudat["time"], mathieudat["Gs"][0], 's')


plt.show()
