import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


class Obj:
    def __init__(self, latent_mean, latent_cov, K):
        self.latent_mean = latent_mean
        self.latent_cov = latent_cov

        self._truncation = K

        cond_var = latent_cov[0, 0]
        cond_var -= latent_cov[0, 1]**2/latent_cov[1, 1]
        cond_sd = np.sqrt(cond_var)

        def _z0_cond(z0, z1):
            cond_mean = latent_mean[0]
            cond_mean += latent_cov[0, 1]*(1./latent_cov[1, 1])*(z1-latent_mean[1])
            return norm.pdf(z0,
                            loc=cond_mean,
                            scale=cond_sd)

        def _z1_marginal(z1):
            return norm.pdf(z1,
                            loc=latent_mean[1],
                            scale=np.sqrt(latent_cov[1, 1]))

        self._z0_cond = _z0_cond
        self._z1_marginal = _z1_marginal

    ###
    # Condition on S1
    def cond_S1(self, S1):

        ks = np.arange(-self._truncation,
                       self._truncation+1,
                       1.)

        ws = self._z1_marginal(S1 + 2*np.pi*ks)
        ws /= np.sum(ws)
        print(ws)

        self._S1 = S1
        self._cond_S1_weights = ws
        
    def z0_cond_S1(self, z0):

        ks = np.arange(-self._truncation,
                       self._truncation+1,
                       1.)

        ps = self._z0_cond(z0, self._S1 + 2*np.pi*ks)
        print(ps)
        return sum(ps*self._cond_S1_weights)


C = np.array([[3.0, 1.5],
              [1.5, 2.0]])
C *= 1.

obj = Obj(np.zeros(2), C, 3)

obj.cond_S1(3.4)
#print(quad(obj.z0_cond_S1, -np.inf, np.inf))

#zz = np.linspace(-6., 6., 100)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(zz, [obj.z0_cond_S1(z) for z in zz], '-')
#plt.show()
