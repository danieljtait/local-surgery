import numpy as np
from gpode.latentforcemodels import VariationalMLFM2
from scipy.integrate import quad
from scipy.special import erf
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

tt = np.linspace(0., 2.1, 6)
yy = tt.copy()

A0 = np.array([[0., 0.], [0., 0.]])
A1 = np.array([[0., -1.], [1., 0.]])
A2 = np.array([[ 0., 0.], [0., 1.]])

vobj = VariationalMLFM2(1, tt, yy, 0.25,
                        As=[A0, A1, A2],
                        lscales=[1.5, 10.],
                        obs_noise_priors=[[3, 0.5],
                                          [3, 0.5]])

print(vobj._cond_vars[0])
