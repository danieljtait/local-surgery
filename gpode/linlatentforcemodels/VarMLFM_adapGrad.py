import numpy as np
from gpode.kernels import GradientMultioutputKernel


class VarMLFM_adapGrad:

    def __init__(self,
                 ModelMatrices,
                 data_times,
                 data_Y,
                 sigmas,
                 gammas,
                 x_gp_pars):

        _x_kernel_type = "sqexp"

        # Attach the data
        self.data_times = data_times
        self.data_Y = data_Y

        # Model matrices
        self.A = np.asarray(ModelMatrices)

        self._R = self.A.shape[0] - 1       # R is the number of latent forces
                                            # there is an additional constant matrix
                                            # A[0]

        self._K = self.A.shape[1]           # Observation dimension
        self._N = data_times.size           # Number of observations
        
        assert(self._K == self.A.shape[2])  # Squareness check

        ###
        # For each component k we attach a GradientMultioutputKernel
        if _x_kernel_type == "sqexp":
            _x_kernels = []
            for kp in x_gp_pars:
                _x_kernels.append(GradientMultioutputKernel.SquareExponKernel(kp))

            self._x_kernels = _x_kernels

        # pseudo-noise variable governing the contribution from the
        # gradient exper
        self._gammas = gammas

        # noise variables
        self._sigmas = sigmas 

        
        ##
        # Attach the gradient kernel covariance objects to the class
        # - the characteristic length scale parameters (those not possessing
        #   tractable marginals, will typically not be changed during a call to
        #   fit 
        _store_gpdx_covs(self)


"""
Model Setup Utility Functions
"""
def _store_gpdx_covs(mobj, missing_data=False):
    mobj.Lxx = []
    mobj.Cxdx = []
    mobj.S_chol = []

    if missing_data:
        tt = mobj.full_times
    else:
        tt = mobj.data_time

    gammas = mobj._gammas

    for k in range(mobj._K):

        kern = mobj._x_kernels[k]

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)

        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))
        I = np.diag(np.ones(Cdxdx_x.shape[0]))
        S = Cdxdx_x + gammas[k]**2*I
        S_chol = np.linalg.cholesky(S)

        mobj.Lxx.append(Lxx)
        mobj.Cxdx.append(Cxdx)
        mobj.S_chol.append(S_chol)