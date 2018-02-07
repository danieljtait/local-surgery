import numpy as np
from gpode.bayes import Parameter, ParameterCollection
from gpode.kernels import Kernel


class VariationalMLFM:

    def __init__(self,
                 gkp,
                 x_kernel_pars=None,
                 gammas=None,
                 sigmas=None, 
                 As=None,
                 data_time=None,
                 data_Y=None,
                 x_kernel_type="sqexp"):

        if isinstance(sigmas, np.ndarray):
            _sigmas = []
            for i, sval in enumerate(sigmas):
                sigpar = Parameter("sigma_{}".format(i))
                sigpar.value = sval
                _sigmas.append(sigpar)

            self._sigmas = ParameterCollection(_sigmas)

        if isinstance(gammas, np.ndarray):
            _gammas = []
            for i, gval in enumerate(gammas):
                gpar = Parameter("gamma_{}".format(i))
                gpar.value = gval
                _gammas.append(gpar)

            self._gammas = ParameterCollection(_gammas)

        if all(isinstance(kp, np.ndarray) for kp in x_kernel_pars):
            if x_kernel_type == "sqexp":
                _x_kernels = []
                for kp in x_kernel_pars:
                    _x_kernels.append(Kernel.SquareExponKernel(kp))
                
                self._x_kernels = _x_kernels

        self._As = np.array(As)


"""
Utility Functions
"""

# Already defined in mlfm adap grad
def _store_gpdx_covs(mobj):
    mobj.Lxx = []
    mobj.Cxdx = []
    mobj.S_chol = []

    tt = mobj.data.time

    for k in range(mobj.K):

        kern = mobj._x_kernels[k]

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)
        
        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Cxdx))
        I = np.diag(np.ones(Cdxdx_x.shape[0]))
        S = Cdxdx_x + mobj.gammas[k].value**2*I
        S_chol = np.linalg.cholesky(S)

        mobj.Lxx.append(Lxx)
        mobj.Cxdx.append(Cxdx)
        mobj.S_chol.append(S_chol)
        

def _get_A_ik(i, k, A, N):
    return np.column_stack([np.diag(a[k, i]*np.ones(N))
                            for a in A[1:, :, :]])

def _get_C_ij(i, j, C, N):
    return C[i*N:(i+1)*N, j*N:(j+1)*N]

def _get_mu_i(i, mu, N):
    return mu[i*N:(i+1)*N]

def _parse_component_k_for_g(k, A, muX, SigmaX,
                             Sk, Ps, N):
    R = A.shape[0]-1

    _Q = np.zeros((R*N, R*N))
    _b = np.zeros(R*N)

    muxk = _get_mu_i(k, mux, N)

    for i in range(K):

        Ai = _get_A_ik(i, k, A, N)
        muxi = _get_mu_i(i, mux, N)
        dmuxi = np.diag(muxi)

        Cki = _get_C_ij(k, i, cov, N)

        b += np.dot(Ai.T, np.diag(np.dot(Sk, np.dot(Ps[k], Cki))))
        b += np.dot(Ai.T, np.dot(np.diag(muxi), np.dot(S, np.dot(Ps[k], muxk))))

        vec_j = np.zeros(N)

        for j in range(K):
            Aj = _get_A_ik(j, k, A, N)
            muxj = _get_mu_i(j, mux, N)

            Cij = _get_C_ij(i, j, cov, N)
            Kij = Sk*Cij + np.dot(dmuxi, np.dot(Sk, np.diag(muxj)))

            Q += np.dot(Ai.T, np.dot(Kij, Aj))
            vec_j += np.dot(Kij, -A[0, k, j]*np.ones(N))

        b += np.dot(Ai.T, vec_j)

        return np.linalg.solve(Q, b), Q
    
##
# Backsub
#
def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))
