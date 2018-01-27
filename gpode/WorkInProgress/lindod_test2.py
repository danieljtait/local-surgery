import numpy as np
from lindod_src import _x_gp_pars
import gpode.kernels


dxkern = gpode.kernels.GradientMultioutputKernel.SquareExponKernel([1., 5.])

tt = np.linspace(0., 2., 5.)
X = np.random.normal(size=tt.size).reshape(tt.size, 1)


phi = [1., 1.]

_x_gp_pars(phi, dxkern, tt, X[:, 0])
