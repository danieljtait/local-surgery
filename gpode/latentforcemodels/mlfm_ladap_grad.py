import numpy as np
from gpode.kernels import GradientMultioutputKernel


class Data:
    def __init__(self, t, Y):
        self.time = t
        self.Y = Y


###########################################################
#                                                         #
# Fits the multiplicative linear force model              #
#                                                         #
#                                                         #
#               __R                                       #
#       dX /    \           (r)                           #
#         /dt = /_  g_r(t) A   x(t),                      #
#                r=0                                      #
#                                                         #
# where g_0(t) = 1 and g_r(t) ~ GP(0,k(t,t')) by using    #
# a version of the AdaptiveGradient matching approach     #
# in Dondlinger et. al adapted for this special case in   #
# which many of the conditional posteriors reduce to      #
# Gaussian random variables                               #
###########################################################
class MulLatentForceModel_adapgrad:
    def __init__(self,
                 xkp,
                 sigmas, gammas,
                 lforce_ktype="sqexp",
                 As=None,
                 data_time=None,
                 data_Y=None):

        self.data = Data(data_time, data_Y)
        self.sigmas = sigmas
        self.gammas = gammas

        # kernel function for the GP modelling the xk-th trajectory
        self._x_kernels = []
        if lforce_ktype == "sqexp":
            for kpar in xkp:
                kern = GradientMultioutputKernel.SquareExponKernel(kpar)
                self._x_kernels.append(kern)
        else:
            raise NotImplementedError

        self._init_parameters()

    def _init_parameters(self):
        self._init_sigmas()
        self._init_gammas()
        self._init_latent_x_kpar()

        # Attach the gradient cov matrices
        _store_gpdx_covs(self)

    def _init_sigmas(self, strategy="prior"):
        for s in self.sigmas:
            s.value = s.prior.rvs()

    def _init_gammas(self, strategy="prior"):
        for g in self.gammas:
            g.value = g.prior.rvs()

    def _init_latent_x_kpar(self):
        for kern in self._x_kernels:
            rv = kern.kpar.prior.rvs()
            for p, x in zip(kern.kpar.parameters.values(),
                            rv):
                p.value = x

"""
Functions describing the model
"""


def _log_eq20_k(xk, fk, mk,
                Lxx, dCd_x, gamma_k,
                phi_k_val=None, phi_k_prior=None):
    exp_arg = _norm_quad_form(xk, Lxx)

    S = dCd_x + gamma_k**2*np.diag(np.ones(xk.size))
    dLd_x = np.linalg.cholesky(S)

    exp_arg += _norm_quad_form(fk-mk, dLd_x)

    if phi_k_prior is not None:
        exp_arg += np.log(phi_k_prior.pdf(phi_k_val))

    return exp_arg


"""
Some utility functions
"""


##
# Backsub
#
def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))


##
# Attaches the covariance function of
#
# [] we should probably store the cholesky decomposition
#    of Cdxdx_x + gamma_k**2*I
def _store_gpdx_covs(mobj):
    mobj.Lxx = []
    mobj.Cxdx = []
    mobj.S_chol = []
#    mobj.Cdxdx_x = []

    tt = mobj.data.time
    for k in range(mobj.data.Y.shape[1]):
        kern = mobj._x_kernels[k]

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)
        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))
        I = np.diag(np.ones(Cdxdx_x.shape[0]))
        S = Cdxdx_x + np.diag(mobj.gammas[k].value**2*I)
        S_chol = np.linalg.cholesky(S)

        mobj.Lxx.append(Lxx)
        mobj.Cxdx.append(Cxdx)
        mobj.S_chol.append(S_chol)


###
# returns the value of the quad. form
#
#    - 0.5 x^T C^{-1} x
#
# that appears in the argument of the exponential in a mvt
# norm pdf using the cholesky decomposition
####
def _norm_quad_form(x, L):
    return -0.5*np.dot(x, np.linalg.solve(L.T, np.linalg.solve(L, x)))
