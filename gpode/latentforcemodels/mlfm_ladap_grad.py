import numpy as np
from gpode.kernels import Kernel, GradientMultioutputKernel
from scipy.stats import norm


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
                 gkp,
                 sigmas, gammas,
                 lforce_ktype="sqexp",
                 As=None,
                 data_time=None,
                 data_Y=None):

        self.data = Data(data_time, data_Y)
        self.N = self.data.Y.shape[0]
        self.K = self.data.Y.shape[1]

        self._As = As
        if As is not None:
            self.R = len(As)

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

        self._g_kernels = []
        for kpar in gkp:
            kern = Kernel.SquareExponKernel(kpar)
            self._g_kernels.append(kern)

        self._init_parameters()

    def gibbs_update_step(self):
        for i in range(self.K):
            self.gibbs_update_xi(i)

        for r in range(1, self.R):
            self.gibbs_update_gr(r)

        for i in range(self.K):
            self.gibbs_update_sigmai(i)

        for i in range(self.K):
            try:
                self.gibbs_update_phi_i(i)
            except:
                pass

        for i in range(self.K):
            try:
                self.gibbs_update_gamma_i(i)
            except:
                pass

    def gibbs_update_xi(self, i):
        cond_mean, cond_cov = self._get_xi_conditional(i)
        L = np.linalg.cholesky(cond_cov)
        xi_rv = np.dot(L, np.random.normal(size=L.shape[0])) + cond_mean
        self._X[:, i] = xi_rv

    def gibbs_update_gr(self, r):

        assert(r >= 1)

        cond_mean, cond_cov = self._get_gr_conditional(r)
        L = np.linalg.cholesky(cond_cov)
        gr_rv = np.dot(L, np.random.normal(size=L.shape[0])) + cond_mean
        self._Gs[r] = gr_rv

    def gibbs_update_sigmai(self, i):
        # Metropolis-within-gibbs update
        def _logp(s):
            lp = np.sum(norm.logpdf(self.data.Y[:, i],
                                    loc=self._X[:, i],
                                    scale=s))
            lprior = self.sigmas[i].prior.logpdf(s)
            return lp + lprior

        sigma_i = self.sigmas[i]
        snew = sigma_i.proposal.rvs(sigma_i.value)

        A = np.exp(_logp(snew) - _logp(sigma_i.value))

        # Check proposal is symmetric

        if np.random.uniform() <= A:
            sigma_i.value = snew

    def gibbs_update_psi_r(self, r):
        pass

    def gibbs_update_phi_i(self, i):
        k_i = self._x_kernels[i]
        phi_i = k_i.kpar

        tt = self.data.time
        xi = self._X[:, i]
        fi = self._dXdt(self._X, self._Gs)[:, i]

        def _l(phi_val):

            Cxx = k_i.cov(0, 0, tt, tt, kpar=phi_val)
            Lxx = np.linalg.cholesky(Cxx)
            Cxdx = k_i.cov(0, 1, tt, tt, kpar=phi_val)
            Cdxdx = k_i.cov(1, 1, tt, kpar=phi_val)

            Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))
            I = np.diag(np.ones(Cdxdx.shape[0]))
            S = Cdxdx_x + self.gammas[i].value**2*I
            S_chol = np.linalg.cholesky(S)

            mi = np.dot(Cxdx.T, _back_sub(Lxx, xi))

            lval = _norm_quad_form(fi-mi, S_chol)
            lprior = phi_i.prior.logpdf(phi_val)
            return lval + lprior

        phi_i_new_val = phi_i.proposal.rvs(phi_i.value())

        A = np.exp(_l(phi_i_new_val) - _l(phi_i.value()))
        if np.random.uniform() <= A:
            for x, p in zip(phi_i_new_val, phi_i.parameters.values()):
                if isinstance(x, float):
                    p.value = x
                else:
                    p.value = x[0]

            # If we accept then the kernel parameters need to be updated
            Cxx = k_i.cov(0, 0, tt, tt)
            Lxx = np.linalg.cholesky(Cxx)
            Cxdx = k_i.cov(0, 1, tt, tt)
            Cdxdx = k_i.cov(1, 1, tt)

            Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))
            I = np.diag(np.ones(Cdxdx.shape[0]))
            S = Cdxdx_x + self.gammas[i].value**2*I
            S_chol = np.linalg.cholesky(S)

            self.Lxx[i] = Lxx
            self.Cxdx[i] = Cxdx
            self.S_chol[i] = S_chol

    def gibbs_update_gamma_i(self, i):
        tt = self.data.time

        k_i = self._x_kernels[i]
        Lxx = self.Lxx[i]
        Cxdx = self.Cxdx[i]
        Cdxdx = k_i.cov(1, 1, tt, tt)

        Cdxdx_x = Cdxdx - np.dot(Cxdx.T, _back_sub(Lxx, Cxdx))

        xi = self._X[:, i]
        mi = np.dot(Cxdx.T, _back_sub(Lxx, xi))
        fi = self._dXdt(self._X, self._Gs)[:, i]

        gamma_i = self.gammas[i]

        def _l(gamma_val):

            S = Cdxdx_x + gamma_val**2*np.diag(np.ones(tt.size))
            S_chol = np.linalg.cholesky(S)

            lv = _norm_quad_form(fi-mi, S_chol)
            lprior = gamma_i.logpdf(gamma_val)

            return lv + lprior

        gamma_i_new_val = gamma_i.proposal(gamma_i.value)

        A = np.exp(_l(gamma_i_new_val) - _l(gamma_i.value))
        if np.random.uniform() <= A:

            S = Cdxdx_x + gamma_i_new_val**2*np.diag(np.ones(tt.size))
            S_chol = np.linalg.cholesky(S)
            self.S_chol[i] = S_chol

    def _init_parameters(self):
        self._init_sigmas()
        self._init_gammas()
        self._init_latent_x_kpar()
        self._init_latent_g_kpar()

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

    def _init_latent_g_kpar(self):
        for kern in self._g_kernels:
            rv = kern.kpar.prior.rvs()
            for p, x in zip(kern.kpar.parameters.values(),
                            rv):
                p.value = x

    def _get_xi_conditional(self, i):
        ms = []
        inv_covs = []
        for k in range(self.K):
            m, ci = self._parse_component_k_for_xi(i, k, True)
            ms.append(m)
            inv_covs.append(ci)

        # We also need to add the prior contribution term for xi
        Li = self.Lxx[i]
        ci = _back_sub(Li, np.diag(np.ones(Li.shape[0])))
        m = np.zeros(Li.shape[0])
        ms.append(m)
        inv_covs.append(ci)

        # And finally the contribution from the data
        m = self.data.Y[:, i]
        ci = np.diag(1./self.sigmas[i].value**2 * np.ones(self.N))

        ms.append(m)
        inv_covs.append(ci)

        mean, cov = _prod_norm_pars(ms, inv_covs)
        return mean, cov

    def _get_gr_conditional(self, r):
        ms = []
        inv_covs = []
        for k in range(self.K):
            m, ci = self._parse_component_k_for_gr(r, k, True)
            if m is None:
                pass
            else:
                ms.append(m)
                inv_covs.append(ci)

        # Contribution from the prior
        Cgg = self._g_kernels[r-1].cov(self.data.time)

        ms.append(np.zeros(Cgg.shape[0]))
        inv_covs.append(np.linalg.inv(Cgg))

        return _prod_norm_pars(ms, inv_covs)

    ##
    # Should be a cleaner way of doing this, something like
    #
    # sum_r [ g_r * <A_r,x> for x in X ]
    #
    def _dXdt(self, X, Gs):

        if Gs is None:
            Gs = self._Gs

        F = []
        for k in range(self.K):
            vv = []
            for j in range(self.K):
                vj = np.sum([self._As[r][k, j]*g
                             for r, g in enumerate(Gs)], axis=0)
                vv.append(vj)
            fk = np.sum([v*x for x, v in zip(X.T, vv)], axis=0)
            F.append(fk)
        return np.array(F).T

    def _log_eq20(self, X=None, Gs=None):
        if X is None:
            X = self._X
        if Gs is None:
            Gs = self._Gs

        F = self._dXdt(X, Gs)

        exp_arg = 0.
        for k in range(self.K):
            Lxx = self.Lxx[k]
            Cxdx = self.Cxdx[k]
            mk = np.dot(Cxdx.T, _back_sub(Lxx, X[:, k]))

            exp_arg += self._log_eq20_k(X[:, k], F[:, k], mk, k)
        return exp_arg

    def _log_eq20_k(self, xk, fk, mk, k):
        # Component from the GP prior on xk
        val1 = _norm_quad_form(xk, self.Lxx[k])

        # Component from the gradient expert
        val2 = _norm_quad_form(fk - mk, self.S_chol[k])

        return val1 + val2

    def _parse_component_k_for_xi(self, i, k, ret_inv=False):
        return _parse_component_k_for_xi(self, i, k, ret_inv)

    def _parse_component_k_for_gr(self, r, k, ret_inv=False):
        return _parse_component_k_for_gr(self, r, k, ret_inv)


"""
Functions describing the model
"""


def _log_eq20_k(xk, fk, mk,
                Lxx, S_chol,
                phi_k_val=None, phi_k_prior=None):
    exp_arg = _norm_quad_form(xk, Lxx)

    exp_arg += _norm_quad_form(fk-mk, S_chol)

    if phi_k_prior is not None:
        exp_arg += np.log(phi_k_prior.pdf(phi_k_val))

    return exp_arg


"""
Some utility functions
"""


###
# For the component k
def _parse_component_k_for_xi(mobj, i, k, ret_inv=False):

    Lxx = mobj.Lxx[k]
    Cxdx = mobj.Cxdx[k]
    S_chol = mobj.S_chol[k]

    vv = []
    for j in range(mobj.data.Y.shape[1]):
        vj = np.sum([mobj._As[r][k, j]*g
                     for r, g in enumerate(mobj._Gs)], axis=0)
        vv.append(vj)

    if i != k:
        xk = mobj._X[:, k]

        da = np.diag(vv[i])
        mk = np.dot(Cxdx.T, _back_sub(Lxx, xk))

        b = mk - np.sum([v*mobj._X[:, j] for j, v in enumerate(vv)
                         if j != i], axis=0)

        cov_inv = np.dot(da, _back_sub(S_chol, da))

        mean = np.dot(np.linalg.pinv(da), b)
        if ret_inv:
            return mean, cov_inv

        else:
            return mean, np.linalg.inv(cov_inv)

    else:
        I = np.diag(np.ones(Lxx.shape[0]))
        T = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, I)))

        b = - np.sum([v*mobj._X[:, j] for j, v in enumerate(vv)
                      if j != i], axis=0)

        A = np.diag(vv[i]) - T

        cov_inv = np.dot(A.T, _back_sub(S_chol, A))
        w1 = np.dot(b, _back_sub(S_chol, A))

        mean = np.linalg.solve(cov_inv, w1)

        if ret_inv:
            return mean, cov_inv
        else:
            return mean, np.linalg.pinv(cov_inv)


def _parse_component_k_for_gr(mobj, r, k, ret_inv=True):

    Lxx = mobj.Lxx[k]
    Cxdx = mobj.Cxdx[k]
    S_chol = mobj.S_chol[k]

    vv = []
    for s in range(mobj.R):
        vs = np.sum([mobj._As[s][k, j]*x
                     for j, x in enumerate(mobj._X.T)], axis=0)
        vv.append(vs)

    if np.all(vv[r] == 0):
        return None, None

    else:
        da = np.diag(vv[r])
        mk = np.dot(Cxdx.T, _back_sub(Lxx, mobj._X[:, k]))
        b = mk - np.sum([v*mobj._Gs[s] for s, v in enumerate(vv)
                         if s != r], axis=0)

        cov_inv = np.dot(da, _back_sub(S_chol, da))
        mean = np.linalg.solve(da, b)

        if ret_inv:
            return mean, cov_inv
        else:
            return mean, np.linalg.inv(cov_inv)


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
        S = Cdxdx_x + mobj.gammas[k].value**2*I
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
