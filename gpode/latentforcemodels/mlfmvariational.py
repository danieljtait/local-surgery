import numpy as np
from scipy.stats import norm
from scipy.linalg import block_diag
from gpode.bayes import Parameter, ParameterCollection
from gpode.kernels import (GradientMultioutputKernel,
                           Kernel)


class Data:
    def __init__(self, t, Y):
        self.time = t
        self.Y = Y


class VariationalMLFM:

    def __init__(self,
                 x_kernel_pars=None,
                 g_kernel_pars=None,
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
                    _x_kernels.append(GradientMultioutputKernel.SquareExponKernel(kp))
                
                self._x_kernels = _x_kernels

        if all(isinstance(kp, np.ndarray) for kp in g_kernel_pars):
            _g_kernels = []
            for kp in g_kernel_pars:
                _g_kernels.append(Kernel.SquareExponKernel(kp))
            self._g_kernels = _g_kernels

            
        self._As = np.array(As)
        self.R = len(As) - 1
        
        if data_Y is not None:
            self.data = Data(data_time, data_Y)
            self.K = data_Y.shape[1]
            self.N = data_time.size

        try:
            self._store_gpg_covs()
        except:
            pass

    def _parse_component_k_for_g(self, k, mux, Sxx):

        N = self.N
        I = np.diag(np.ones(N))
        
        Skinv = _back_sub(self.S_chol[k], I)
        Pk = np.dot(self.Cxdx[k].T, _back_sub(self.Lxx[k], I))
        
        mk, cov_invk = _parse_component_k_for_g(k, self._As, mux, Sxx,
                                                Skinv, Pk, N)

        """
        print(mk)
        def _obj_func(g):
            G = np.concatenate((np.ones(self.N),
                                g)).reshape(self.R+1, self.N)
            return _comp_k(k, G, Skinv, self._As, Pk, mux, Sxx, N)

        from scipy.optimize import minimize
        res = minimize(_obj_func, np.ones(N*self.R))
        print(res.x)
        assert(np.all(mk == res.x))
        """
        return mk, cov_invk

    def _parse_component_k_for_x(self, k, mug, Sgg):

        N = self.N
        I = np.diag(np.ones(N))
        
        Skinv = _back_sub(self.S_chol[k], I)
        Pk = np.dot(self.Cxdx[k].T, _back_sub(self.Lxx[k], I))

        return hfunc(k, Skinv, Pk, mug, Sgg,
                     self._As, self.N, self.K, self.R)
        """
        sigmas = self._sigmas.value()

        def _obj_func(x):
            X = x.reshape(self.K, self.N)
            val1 = _comp_k_for_x(k, X, Skinv, self._As, Pk, mug, Sgg, self.N, self.R)
            val2 = 0
#            val2 = np.sum([norm.logpdf(self.data.Y[:, k], loc=X[k, :], scale=sigmas[k])
#                           for k in range(self.K)])
            return -(val1+val2)

        x0 = self.data.Y.T.ravel()

        from scipy.optimize import minimize
        res = minimize(_obj_func, x0)
        print("=======================")
        print(np.linalg.inv(res.hess_inv))
        print("=======================")        
#        print(_obj_func(np.zeros(res.x.size)) <= res.fun)
        return res.x, np.linalg.inv(res.hess_inv)
    """

    def _store_gpdx_covs(self):
        _store_gpdx_covs(self)

    def _store_gpg_covs(self):
        _store_gpg_covs(self)

    def _get_g_conditional(self, mux, Sxx):
        ms = []
        cinvs = []
        for k in range(self.K):
            try:
                m, ci = self._parse_component_k_for_g(k, mux, Sxx)
                ms.append(m)
                cinvs.append(ci)
            except:
                pass

        ci_prior = block_diag(*(cinv for cinv in self.Cgg_inv))
        mean_prior = np.zeros(ci_prior.shape[0])
        print(np.linalg.norm(cinvs[0]))
        print(np.linalg.norm(ci_prior))
        ms.append(mean_prior)
        cinvs.append(ci_prior)
        
        mean, cov = _prod_norm_pars(ms, cinvs)
        return mean, cov

    def _get_x_conditional(self, mug, Sgg):
        ms = []
        cinvs = []

        for k in range(self.K):
            try:
                m, ci = self._parse_component_k_for_x(k, mug, Sgg)
                ms.append(m)
                cinvs.append(ci)
            except:
                print("Sad face!")
                pass

        ms.append(self.data.Y.T.ravel())
        sigmas = self._sigmas.value()
        In = np.ones(self.N)
        ci_data = np.diag(np.concatenate((In/sigmas[0]**2, In/sigmas[1]**2)))
        cinvs.append(ci_data)

        C1inv = _back_sub(self.Lxx[0], np.diag(np.ones(self.N)))
        C2inv = _back_sub(self.Lxx[1], np.diag(np.ones(self.N)))
        ci = block_diag(C1inv, C2inv)
        mi = np.zeros(ci.shape[0])
        ms.append(mi)
        cinvs.append(ci)

        mean, cov = _prod_norm_pars(ms, cinvs)
        return mean, cov

    def func(self, k, mg, Cgg):
        I = np.diag(np.ones(self.N))
        Skinv = _back_sub(self.S_chol[k], I)
        Pk = np.dot(self.Cxdx[k].T, _back_sub(self.Lxx[k], I))        

        hfunc(k, Skinv, Pk, mg, Cgg, self._As, self.N, self.K, self.R)

        self._parse_component_k_for_x(k, mg, Cgg)

#        _a_func()
#        X = self.data.Y.T

        
#        func(k, X, self._As, self.R, mg, Cgg, Skinv)
#        func2(k, X, self._As, self.R, mg, Cgg, Skinv)        
            
"""
Utility Functions
"""

# Already defined in mlfm adap grad
def _store_gpdx_covs(mobj):
    mobj.Lxx = []
    mobj.Cxdx = []
    mobj.S_chol = []

    tt = mobj.data.time
    gammas = mobj._gammas.value()
    for k in range(mobj.K):

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

def _store_gpg_covs(mobj):
    mobj.Cgg_inv = []

    tt = mobj.data.time
    I = np.diag(np.ones(tt.size))
    
    for kern in mobj._g_kernels:

        Cgg = kern.cov(tt)
        Lgg = np.linalg.cholesky(Cgg)
        mobj.Cgg_inv.append(_back_sub(Lgg, I))
        
        

def _get_A_ik(i, k, A, N):
    return np.column_stack([np.diag(a[k, i]*np.ones(N))
                            for a in A[1:, :, :]])

def _get_C_ij(i, j, C, N):
    return C[i*N:(i+1)*N, j*N:(j+1)*N]

def _get_mu_i(i, mu, N):
    return mu[i*N:(i+1)*N]

def _parse_component_k_for_g(k, A, muX, cov,
                             Sk, Pk, N):
    K = A[0].shape[0]
    R = A.shape[0]-1

    _Q = np.zeros((R*N, R*N))
    _b = np.zeros(R*N)

    muxk = _get_mu_i(k, muX, N)

    for i in range(K):

        Ai = _get_A_ik(i, k, A, N)
        muxi = _get_mu_i(i, muX, N)
        dmuxi = np.diag(muxi)

        Cki = _get_C_ij(k, i, cov, N)

        _b += np.dot(Ai.T, np.diag(np.dot(Sk, np.dot(Pk, Cki))))
        _b += np.dot(Ai.T, np.dot(np.diag(muxi), np.dot(Sk, np.dot(Pk, muxk))))

        vec_j = np.zeros(N)

        for j in range(K):
            Aj = _get_A_ik(j, k, A, N)
            muxj = _get_mu_i(j, muX, N)

            Cij = _get_C_ij(i, j, cov, N)
            Kij = Sk*Cij + np.dot(dmuxi, np.dot(Sk, np.diag(muxj)))

            _Q += np.dot(Ai.T, np.dot(Kij, Aj))
            vec_j += np.dot(Kij, -A[0, k, j]*np.ones(N))

        _b += np.dot(Ai.T, vec_j)
        return np.linalg.solve(_Q, _b), _Q
    
##
# Backsub
#
def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))



"""
Functions for testing, delete
"""
def _get_vi_k(i, k, G, A):
    return np.dot(A[:, k, i], G)

def _comp_k(k, G, S, A, Pk, mux, cov, N):
    K = A[0].shape[0]
    val = 0
    for i in range(K):
        vi = _get_vi_k(i, k, G, A)
        dvi = np.diag(vi)
        mi = _get_mu_i(i, mux, N)
        if i == k:
            dvi -= Pk.T

        for j in range(K):
            vj = _get_vi_k(j, k, G, A)
            dvj = np.diag(vj)

            if j == k:
                dvj -= Pk

            Cji = _get_C_ij(j, i, cov, N)
            M = np.dot(dvi, np.dot(S, np.dot(dvj, Cji)))

            val1 = np.trace(M)

            Q = np.dot(dvi, np.dot(S, dvj))
            mj = _get_mu_i(j, mux, N)

            val2 = np.dot(mi, np.dot(Q, mj))

            val += val1 + val2
    return 0.5*val
                         
def _get_wr_k(r, k, X, A):
    return np.dot(A[r, k, :], X)

def _get_Wk(k, X, A, R):
    Wk = np.column_stack([np.diag(_get_wr_k(r, k, X, A))
                          for r in range(R+1)])
    return Wk

def _comp_k_for_x(k, X, S, A, Pk, mug, cov, N, R):

    K = A[0].shape[0]

    Wk = _get_Wk(k, X, A, R)

    Q = np.dot(Wk[:, N:], np.dot(cov, Wk[:, N:].T))

    mug = np.concatenate((np.ones(N), mug)).reshape(R+1, N)
    meta = np.dot(Wk.T, mug.T)

    meta = np.sum([_get_wr_k(r, k, X, A)*mg for r, mg in enumerate(mug)],
                  axis=0)
    mk = np.dot(Pk, X[k, ])
    meta -= mk

    val = np.trace(np.dot(Q, S)) + np.dot(meta, np.dot(S, meta))

    """


    res = np.zeros((N, N))
    for s in range(R):
        wi = _get_wr_k(s+1, k, X, A)
        dwi = np.diag(wi)
        for t in range(R):
            wj = _get_wr_k(t+1, k, X, A)
            dwj = np.diag(wj)

            Cst = _get_C_ij(s, t, cov, N)
            
            res += np.dot(dwi, np.dot(Cst, dwj))

    print("=========")
    print(res)
    """
    return -0.5*val


def func(k, X, A, R, mu, cov, Sk):

    N = X.shape[1]
    K = A[0].shape[0]
    I = np.diag(np.ones(N))

    def _u_r_k(r, k):
        a = A[r, k, :]
        return np.dot(a, X)

    val = 0.
    val2 = 0.

    _M = []
    for r in range(R+1):
        m = np.column_stack([a*I for a in A[r, k, :]])
        _M.append(m)
    
    val3 = 0.
    val4 = np.zeros((X.size, X.size))

    res = 0.
    mu = np.concatenate((np.ones(N), mu))
    
    for s in range(R+1):
        mus = _get_mu_i(s, mu, N)
        us = _u_r_k(s, k)
        dus = np.diag(us)
        
        for t in range(R+1):
            mut = _get_mu_i(t, mu, N)
            ut = _u_r_k(t, k)
            dut = np.diag(ut)

            if s > 0 and t > 0:
                Cts = _get_C_ij(t-1, s-1, cov, N)
                M = np.dot(dus, np.dot(Sk, dut))
                val += np.trace(np.dot(M, Cts))
                val2 += np.dot(us, np.dot(Sk * Cts.T, ut))

                val3 += np.dot(np.dot(_M[s], X.ravel()),
                               np.dot(Sk * Cts.T,
                                      np.dot(_M[t], X.ravel())))

                _S = np.dot(_M[s].T, np.dot(Sk * Cts.T, _M[t]))
                val4 += _S

            res += np.dot(mus*us, np.dot(Sk, mut*ut))

    val4 = np.dot(X.ravel(), np.dot(val4, X.ravel()))
    print(val, val2, val3, val4)
    print(val + res)


def func2(k, X, A, R, mu, cov, Sk):

    N = X.shape[1]
    K = A[0].shape[0]
    I = np.diag(np.ones(N))

    mu = np.concatenate((np.ones(N), mu))

    _M = []
    for r in range(R+1):
        m = np.column_stack([a*I for a in A[r, k, :]])
        _M.append(m)

    K = np.zeros((N*K, N*K))

    val = 0.
    
    for s in range(R+1):
        mus = _get_mu_i(s, mu, N)
        dmus = np.diag(mus)

        for t in range(R+1):
            mut = _get_mu_i(t, mu, N)
            dmut = np.diag(mut)
            

            _Kst = np.dot(dmus, np.dot(Sk, dmut))
            _Kst = np.dot(_M[s].T, np.dot(_Kst, _M[t]))
            K += _Kst

            if s > 0 and t > 0:
                Cts = _get_C_ij(t-1, s-1, cov, N)
                
                _Kst = np.dot(_M[s].T, np.dot(Sk * Cts.T, _M[t]))
                K += _Kst
                val += np.dot(X.ravel(), np.dot(_Kst, X.ravel()))

    print(val)
    print(np.dot(X.ravel(), np.dot(K, X.ravel())))

def _a_func():
    N = 3
    K = 2
    R = 3

    k = 1

    S = np.random.normal(size=N*R*N*R).reshape(N*R, N*R)

    A0 = np.random.normal(size=K*K).reshape(K, K)
    A1 = np.random.normal(size=K*K).reshape(K, K)
    A2 = np.random.normal(size=K*K).reshape(K, K)
    A3 = np.random.normal(size=K*K).reshape(K, K)
    A = np.array([A0, A1, A2, A3])

    def _get_M(i, j):
        M = np.zeros((N, N))
        for s in range(R):
            for t in range(R):
                Srs = S[s*N:(s+1)*N, t*N:(t+1)*N]
                M += A[s+1, k, i]*A[t+1, k, j]*Srs
        return M

    C00 = _get_M(0, 0)
    C01 = _get_M(0, 1)
    C10 = _get_M(1, 0)
    C11 = _get_M(1, 1)

    mat = np.row_stack((
        np.column_stack((a*np.diag(np.ones(N)) for a in A[1:,k,i]))
        for i in range(K)))

#    print(mat.shape)
#    print(S.shape)
#    print(A[1:, k, 0])    
    K1 = np.row_stack((np.column_stack((C00, C01)),
                       np.column_stack((C10, C11))))
    K2 = np.dot(mat, np.dot(S, mat.T))
    print(K1)
    print(K2)



def hfunc(k, Skinv, Pk, mug, Cgg, A, N, K, R):
    I_N = np.diag(np.ones(N))

    ###
    # Calculate the covariance matrix of the terms
    #
    #     u_i = Σ_r A[r, k, i] ○ g_r     
    #
    # by transforming the covariance matrix Sgg of g
    # 
    A_transform = np.row_stack((
        np.column_stack((a*I_N for a in A[1:,k, i]))
        for i in range(K)))

    S_uu = np.dot(A_transform, np.dot(Cgg, A_transform.T))

    # Exp_G[r, ] = E[g_r]     
    Exp_G = np.concatenate((np.ones(N), mug)).reshape(R+1, N)

    # Exp_U[i, ] = Σ_r A[r, k, i] ○ E[g_r]     
    Exp_U = np.dot(A[:, k, :], Exp_G)
    diag_Exp_U = np.column_stack((np.diag(row) for row in Exp_U))

    ###
    #
    _S_transform = np.row_stack((
        np.column_stack((Skinv for j in range(K)))
        for i in range(K)))

    # S_k^{-1} ○ C_{u_i, u_j}
    expr0 = _S_transform * S_uu

    # diag(E[u_i]) Skinv diag(E[u_j])
    expr0 += np.dot(diag_Exp_U.T, np.dot(Skinv, diag_Exp_U))

    #
    expr1 = np.zeros(expr0.shape)
    SkinvPk = np.dot(Skinv, Pk)
    expr1[k*N:(k+1)*N, :] = np.column_stack((SkinvPk for k in range(K)))

    expr2 = np.zeros((N, N*K))
    expr2[:, k*N:(k+1)*N] = Pk
    expr2 = np.dot(expr2.T, np.dot(Skinv, expr2))

    cov_inv = expr0 + expr1 + expr1.T + expr2

    return np.zeros(cov_inv.shape[0]), cov_inv

##
#  p(x) ∝ Π N(x, means[k] | inv_covs[k])
def _prod_norm_pars(means, inv_covs):
    m1 = means[0]
    C1inv = inv_covs[0]

    if len(means) == 1:
        return m1, np.linalg.inv(C1inv)

    else:

        for m2, C2inv in zip(means[1:], inv_covs[1:]):
            Cnew_inv = C1inv + C2inv
            mnew = np.linalg.solve(Cnew_inv,
                                   np.dot(C1inv, m1) + np.dot(C2inv, m2))
            m1 = mnew
            C1inv = Cnew_inv

        return mnew, np.linalg.inv(Cnew_inv)
