import numpy as np
from gpode import latentforcemodels as lfm
from gpode.examples import DataLoader
from gpode.bayes import Parameter, ParameterCollection

MLFM = lfm.MulLatentForceModel_adapgrad

tt = np.linspace(0.5, 10., 10)
bd = DataLoader.load("bessel jn", 11, tt, [0.1, 0.05],
                     order=2)
np.random.seed()  # Reseed the generator seeded in DataLoader

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

# Make the obs. noise parameters
sigmas = [Parameter("sigma_{}".format(k),
                    prior=("gamma", (4, 0.2)),
                    proposal=("normal rw", 0.1))
          for k in range(K)]

# Make the grad. noise parameters
gammas = [Parameter("sigma_{}".format(k),
                    prior=("gamma", (4, 0.2)),
                    proposal=("normal rw", 0.1))
          for k in range(K)]

m = MLFM(xkp,
         sigmas, gammas,
         As=bd["As"],
         data_time=bd["time"],
         data_Y=bd["Y"])

m._Gs = [np.ones(bd["time"].size)] + bd["Gs"]

