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

        # kernel function for the GP modelling the xk-th trajectory
        self._x_kernels = []
        if lforce_ktype == "sqexp":
            for kpar in xkp:
                kern = GradientMultioutputKernel.SquareExponKernel(kpar)
                self._x_kernels.append(kern)
        else:
            raise NotImplementedError


"""
Functions describing the model
"""


def _log_eq20_k(xk, fk, mk,
                Lxx, dCd_x, gamma_k,
                phi_k_val=None, phi_k_prior=None):
    exp_arg = _norm_quad_form(xk, Lxx)

    S = dCd_x + gamma_k*np.diag(np.ones(xk.size))
    dLd_x = np.linalg.cholesky(S)

    exp_arg += _norm_quad_form(fk-mk, dLd_x)

    if phi_k_prior is not None:
        exp_arg += np.log(phi_k_prior.pdf(phi_k_val))

    return exp_arg


"""
Some utility functions
"""


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
