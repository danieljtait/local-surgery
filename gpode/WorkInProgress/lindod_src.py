import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint
from scipy.stats import norm, multivariate_normal
from gpode.bayes import Parameter
from gpode.kernels import GradientMultioutputKernel
import gpode.gaussianprocesses
from scipy.optimize import minimize


class Data:
    def __init__(self, time, Y):
        self.time = time
        self.Y = Y


class MGPLinearAdapGrad3:
    def __init__(self,
                 xkp, lforce_ktype="sqexp"):

        # kernel function for the GP modelling the xk-th trajectory
        self._x_kernels = []
        if lforce_ktype == "sqexp":
            for kpar in xkp:
                kern = GradientMultioutputKernel.SquareExponKernel(kpar)
                self._x_kernels.append(kern)
        else:
            raise NotImplementedError

    def model_setup(self, xkernels="prior"):
        if xkernels == "prior":
            for kern in self._x_kernels:
                rv = kern.kpar.prior.rvs()
                for p, x in zip(kern.kpar.parameters.values(), rv):
                    p.value = x


    def _phi_k_mh_update(self, k):
        kern = self._x_kernels[k]
        kpar = kern.kpar

        new_val = kpar.proposal.rvs(kpar.value())

        def _lgp_hyperpar_pdf(val):
            # new cov parameters
            Lxx, mdx_x, Cdxdx_x = _x_gp_pars(new_val,
                                             kern,
                                             self.data.time,
                                             self.data._X[:, k])


## Corresponds to equation (20) in
#
def _norm_quad_form(x, L):
    return -0.5*np.dot(x, np.linalg.solve(L.T, np.linalg.solve(L, x)))


def _log_eq20(X, As, Gs, dXdt,
              gammas, sigmas,
              LCs, dms, dCs, k=-1,
              phi_k_val=None, phi_k_prior=None):

    if k < 0:
        kiter = range(X.shape[1])
    else:
        kiter = [k]

    exp_arg = 0.
    F = dXdt(X, As, Gs)

    for k in kiter:

        # Contribution from the data trajectory expert
        LC = LCs[k]       # chol. decomp of the cov for the data expert
        exp_arg += _norm_quad_form(X[:, k], LC)

        # Contribution for the gradient expert
        fk = F[:, k]
        mk = dms[k]
        etak = fk - mk

        S = dCs[k] + np.diag(gammas[k]*np.ones(dCs[k].shape[0]))
        dLd_x = np.linalg.cholesky(S)
        exp_arg += _norm_quad_form(etak, dLd_x)

    if phi_k_prior is not None:
        exp_arg += np.log(phi_k_prior.pdf(phi_k_val))

    return exp_arg


class MGPLinearAdapGrad2:
    def __init__(self, As, data, sigmas, kpars, lforce_ktype="sqexp"):

        self.R = len(As)
        self.output_dim = data.Y.shape[1]

        self.data = data

        self.As = As          # Basis matrices
        self.Gs = [np.ones(data.time.size)] + [None for r in range(self.R-1)]
        self.sigmas = sigmas  # Observation noise scale

        self.latent_gp_kernels = []
        if lforce_ktype == "sqexp":
            # Each latent force is a Gaussian process with square
            # exponential kernel function with only the inverse
            # length scale parameter varying
            for kpar in kpars:
                kern = GradientMultioutputKernel.SquareExponKernel(kpar)
                self.latent_gp_kernels.append(kern)

        self._LCs = [None for k in range(self.output_dim)]
        self._dmcs = [None for k in range(self.output_dim)]
        self._dCs = [None for k in range(self.output_dim)]

    def model_setup(self, X="gp", kernels="prior", sigmas="prior", Gs="prior"):
        self._init_sigmas(sigmas)
        self._init_latent_gp_kernels(kernels)
        self._init_latent_traj(X)
        self._init_Gs(Gs)

    def _init_latent_gp_kernels(self, strategy):
        if strategy == "prior":
            for k in self.latent_gp_kernels:
                k.kpar.value = k.kpar.prior.rvs()

    def _init_sigmas(self, strategy):
        if strategy == "prior":
            pass

    def _init_latent_traj(self, strategy):
        if strategy == "gp":
            # Initalise the latent trajectory using
            # a Gaussian process fit to the model data
            X = []
            for i, kern in enumerate(self.latent_gp_kernels):
                Cxx = kern.cov(0, 0, self.data.time, self.data.time)
                I = np.diag(np.ones(Cxx.shape[0]))

                ovar = self.sigmas[i]**2
                y = self.data.Y[:, i]

                L = np.linalg.cholesky(Cxx + ovar*I)
                mean = np.dot(Cxx,
                              np.linalg.solve(L.T,
                                              np.linalg.solve(L, y)))

                cov = ovar*np.dot(Cxx,
                                  np.linalg.solve(L.T,
                                                  np.linalg.solve(L, I)))

                xkrv = multivariate_normal.rvs(mean=mean, cov=cov)
                X.append(xkrv)
            self.latent_X = np.column_stack((x for x in X))

    def _init_Gs(self, strategy):
        if strategy == "prior":
            pass
        elif isinstance(strategy, list):
            for r, g in enumerate(strategy):
                self.Gs[1+r] = g

    def _dXdt(self, X, As=None, Gs=None):
        if As is None:
            As = self.As
        if Gs is None:
            Gs = self.latent_force_vals
        if X is None:
            X = self.X

        return np.sum([np.array([np.dot(A*g_, x) for g_, x in zip(g, X)])
                       for A, g in zip(As, Gs)], axis=0)

    def _phi_k_mh_update(self, k):
        kern = self.latent_gp_kernels[k]
        kpar = kern.kpar

        new_val = kpar.proposal.rvs(kpar.value)

        lpcur = lgp_hyperpar_pdf(kpar.value, k, kpar.prior,
                                 kern, self.data.time,
                                 self.As, self.Gs,
                                 self.sigmas, self.gammas,
                                 self.latent_X, self._dXdt)

        lpnew = lgp_hyperpar_pdf(new_val, k, kpar.prior,
                                 kern, self.data.time,
                                 self.As, self.Gs,
                                 self.sigmas, self.gammas,
                                 self.latent_X, self._dXdt)

        A = np.exp(lpnew - lpcur)
        if np.random.uniform() <= A:
            kpar.value = new_val
            self._update_stored_xgp_pars(k, kern)
        else:
            pass

    def _psi_r_mh_update(self, r):
        #############################################
        # conditional on Gs[r] we have a relatively #
        # straight forward update of the prior      #

        kern = self.Gs_kernels[r]

        def _lp(z):
            g = self.Gs[r]
            cov = self.Gs_kernels[r].cov(self.data.time,
                                         self.data.time,
                                         kpar=z)

            return multivariate_normal.logpdf(g,
                                              mean=np.zeros(g.size),
                                              cov=cov)
        lpcur = _lp(kern.kpar)

        new_val = [x + np.random.normal(scale=0.1) for x in kern.kpar]

        try:
            lpnew = _lp(new_val)
            A = np.exp(lpnew - lpcur)
            if np.random.normal() <= A:
                kern.kpar = new_val
            else:
                pass
        except:
            pass

    def _g_r_update(self, r):
        C0 = self.Gs_kernels[r].cov(self.data.time)
        prior = multivariate_normal(mean=np.zeros(C0.shape[0]), cov=C0)

        m, C = gr_post_conditional(r, self.As, self.Gs,
                                   self.sigmas, self.gammas,
                                   self.latent_X,
                                   self._dmcs, self._LCs, self._dCs,
                                   self._dXdt, prior)
        rg = multivariate_normal.rvs(mean=m, cov=C)
        self.Gs[r+1] = rg

    def _x_k_update(self, k):

        try:
            m, C = xk_post_conditional(k, self.As, self.Gs,
                                       self.sigmas, self.gammas,
                                       self.data.Y, self.latent_X,
                                       self._dmcs, self._LCs, self._dCs,
                                       self._dXdt, None)
            rx = multivariate_normal.rvs(mean=m, cov=C)
            self.latent_X[:, k] = rx
        except:
            pass

    def _update_stored_xgp_pars(self, k, kern):
        L, mdx_x, Cdxdx_x = _x_gp_pars(None, kern,
                                       self.data.time, self.latent_X[:, k])
        self._dmcs[k] = mdx_x
        self._LCs[k] = L
        self._dCs[k] = Cdxdx_x
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
# in Dodlinger et. al adapted for this special case in    #
# which many of the conditional posteriors reduce to      #
# Gaussian random variables                               #
###########################################################
class MGPLinearAdapGrad:
    def __init__(self, As, data, latent_gps, latent_force_vals=None):

        self.R = len(As)  # Total number of model matrices
        self.output_dim = data.Y.shape[1]

        self.latent_force_vals = [np.ones(data.time.size)]
        if latent_force_vals is None:
            self.latent_force_vals += [None for i in range(self.R-1)]
        else:
            self.latent_force_vals += latent_force_vals

        self.data = data
        self.latent_gps = latent_gps

        self._latent_gp_grad_mean = {}
        self._latent_gp_grad_cov = {}

        self.As = As

#        self._dxkernels = 

    def init(self):

        ##
        # Initalise latent trajectory with the Gaussian
        # processes estimators based on the data
        for k, gp in enumerate(self.latent_gps):
            gp.fit(y_data=[self.data.time, self.data.Y[:, k]])
        state_update(self.latent_gps, [0.1, 0.1], self.data.Y)
        self.X = np.column_stack((gp.training_data["y"][1]
                                  for gp in self.latent_gps))

    def _dXdt(self, X, As=None, Gs=None):
        if As is None:
            As = self.As
        if Gs is None:
            Gs = self.latent_force_vals
        if X is None:
            X = self.X

        return np.sum([np.array([np.dot(A*g_, x) for g_, x in zip(g, X)])
                       for A, g in zip(As, Gs)], axis=0)

    #############################################
    #                                           #
    # Need to store the mean and and covariance #
    # of the gradient Gaussian processes        #
    #                                           #
    #############################################
    def _store_gp_grad_pars(self):
        for k, gp in enumerate(self.latent_gps):
            m, C = gp.predict(dy_inputs=gp.training_data["y"][0],
                              ret_par=True)
            self._latent_gp_grad_mean[k] = m
            self._latent_gp_grad_cov[k] = C

    ##############################################
    # Parameters of the conditional distribution #
    #                                            #
    #      (r)                                   #
    #     A                                      #
    #      ij                                    #
    #                                            #
    ##############################################
    def _arkj_par(self, r, k, j, As, Gs,
                  X, mk, Ckinv):
        F = np.zeros(X.shape)
        for A, g in zip(As, Gs):
            F += np.array([np.dot(A*g_, x) for g_, x in zip(g, X)])

        mrkj = As[r][k, j]*X[:, j]*Gs[r] - F[:, k] + mk
        var = 1./np.dot(X[:, j], np.dot(Ckinv, X[:, j]))

        return var*np.dot(X[:, j]*Gs[r], np.dot(Ckinv, mrkj)), var

#    def _updf(self, As, Gs, X, dms, dCs):
#        exp_arg = 0.
#        for k in range(self.output_dim):
#            m = dms[k]
#            C = dCs[k]
#
#            exp_arg += 1.
#        return np.exp(exp_arg)


####### Utility functions for the model
#
# The unofmralised probability density function
#
# Eq X in cite ... {}
def log_updf(As, Gs,
             sigmas, gammas,
             X, dms, LCs, dCs,
             dXdt):

    F = dXdt(X, As, Gs)

    exp_arg = 0.
    for k in range(X.shape[1]):
        fk = F[:, k]
        mk = dms[k]
        etak = fk - mk

        LC = LCs[k]

        S = dCs[k] + np.diag(gammas[k]*np.ones(LC.shape[0]))
        LS = np.linalg.cholesky(S)

        val1 = np.dot(X[:, k],
                      np.linalg.solve(LC.T, np.linalg.solve(LC, X[:, k])))
        val2 = np.dot(etak, np.linalg.solve(LS.T, np.linalg.solve(LS, etak)))
        exp_arg -= 0.5*(val1 + val2)

    return exp_arg


def lgp_hyperpar_pdf(val, k, phi_k_prior,
                     kernel, tt,
                     As, Gs,
                     Sigmas, gammas,
                     X, dXdt):

    phi_k = [1., val]

    xk = X[:, k]
    Lxx, mdx_x, Cdxdx_x = _x_gp_pars(phi_k, kernel, tt, xk)
    Ldxdx_x = np.linalg.cholesky(Cdxdx_x)

    fk = dXdt(X, As, Gs)[:, k]
    etak = fk - mdx_x

    expr1 = -0.5*np.dot(xk, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, xk)))
    expr2 = -0.5*np.dot(etak,
                        np.linalg.solve(Ldxdx_x.T,
                                        np.linalg.solve(Ldxdx_x, etak)))
    expr3 = phi_k_prior.logpdf(val)

    return expr1 + expr2 + expr3


###############################################################
#                                                             #
# For a gaussian process xk ~ GP(0, kernel(phi_k)) returns    #  
# the process cov matrix and the parameters of
#
#     dx | x ~ N(mdx_x, Cdxdx_x)
#
##############################################################
def _x_gp_pars(phi_k, kernel, tt, xk):
    Cxx = kernel.cov(0, 0, tt, tt, phi_k)
    Cxdx = kernel.cov(0, 1, tt, tt, phi_k)
    Cdxdx = kernel.cov(1, 1, tt, tt, phi_k)

    Lxx = np.linalg.cholesky(Cxx)

    # _mat = Cdxx Cdxdx^{-1} Cxdx
    _mat = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, Cxdx)))
    Cdxdx_x = Cdxdx - _mat

    mdx_x = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, xk)))

    return Lxx, mdx_x, Cdxdx_x


def gr_post_conditional(r, As, Gs,
                        sigmas, gammas,
                        X, dms, LCs, dCs,
                        dXdt, prior):

    def _objfunc(g):
        Gs_ = [G for G in Gs]
        Gs_[r] = g
        negll = -log_updf(As, Gs_, sigmas, gammas, X,
                          dms, LCs, dCs, dXdt)
        neglprior = -prior.logpdf(g)
        return negll + neglprior

    g0 = np.zeros(X.shape[0])
    res = minimize(_objfunc, g0)
    if res.status != 0:
        print(res.message)
    
    return res.x, res.hess_inv


def xk_post_conditional(k, As, Gs,
                        sigmas, gammas,
                        Y, X,
                        dms, LCs, dCs, dXdt,
                        xk_prior):

    def _objfunc(xk):
        lpobs = np.sum(norm.logpdf(Y[:, k], loc=xk, scale=sigmas[k]))

        X_ = X.copy()
        X_[:, k] = xk

        LC = LCs[k]
        val1 = np.dot(X_[:, k],
                      np.linalg.solve(LC.T, np.linalg.solve(LC, X_[:, k])))

        exp_arg = 0.
        for j in range(X_.shape[1]):
            fj = dXdt(X_, As, Gs)[:, j]
            etaj = dms[j] - fj

            S = dCs[j] + np.diag(gammas[j]*np.ones(LC.shape[0]))
            LS = np.linalg.cholesky(S)

            exp_arg += -0.5*np.dot(etaj,
                                   np.linalg.solve(LS.T,
                                                   np.linalg.solve(LS, etaj)))
        exp_arg += val1

        return -(lpobs + exp_arg)  # + lprior)

    xk0 = X[:, k].copy()
    res = minimize(_objfunc, xk0)
    if res.status != 0:
        print(res.message)
    return res.x, res.hess_inv


def state_update(gps, sigmas, obs_Y):
    for k, gp in enumerate(gps):
        C = np.dot(gp.Ls["y"], gp.Ls["y"].T)
        I = np.diag(np.ones(C.shape[0]))

        L = np.linalg.cholesky(C + sigmas[k]*I)

        mean = np.dot(C, np.linalg.solve(L.T,
                                         np.linalg.solve(L, obs_Y[:, k])))

        cov = sigmas[k]*np.dot(C, np.linalg.solve(L.T,
                                                  np.linalg.solve(L, I)))

        Xrv = multivariate_normal.rvs(mean=mean, cov=cov)
        
        gp.fit(y_data=[gp.training_data["y"][0],
                       Xrv])


class HMC:
    def __init__(self, E, Egrad, eps):
        self.E = E
        self.Egrad = Egrad
        self.eps = eps
        self.momenta_scale = 1.

    def H(self, z, r):
        return self.E(z) + 0.5*sum(r**2)

    def leapfrog_update(self, zcur, rcur, eps, Hcur, *args, **kwargs):

        rhalfstep = rcur - 0.5*eps*self.Egrad(zcur, *args, **kwargs)
        znew = zcur + eps*rhalfstep / self.momenta_scale**2
        rnew = rhalfstep - 0.5*eps*self.Egrad(znew, *args, **kwargs)

        Hnew = self.H(znew, rnew)
        A = np.exp(Hcur - Hnew)

        if np.random.uniform() <= A:
            return znew, rnew, Hnew

        else:
            if abs(Hnew-Hcur) > 0.1*Hcur:
                print("Reject", Hnew, Hcur)
            return zcur, rcur, Hcur

    def momenta_update(self, rcur, *args, **kwargs):
        return np.random.normal(scale=self.momenta_scale,
                                size=rcur.size)

    def sample(self, zcur, n_steps=10, *args, **kwargs):
        rcur = np.random.normal(size=zcur.size)
        eps = np.random.choice([-1., 1.])*self.eps
        Hcur = self.H(zcur, rcur)
        for k in range(n_steps):
            zcur, rcur, Hcur = self.leapfrog_update(zcur, rcur, eps, Hcur)
        return zcur

"""
#np.random.seed(21)
N = 3
K = 2

# The model parameters will be a_{ij}^{r} = [A_r]_{ij}
A = np.random.normal(size=4).reshape(2, 2)
B = np.random.normal(size=4).reshape(2, 2)
C = np.random.normal(size=4).reshape(2, 2)                                     


x0 = np.array([1., 0.])
tt = np.linspace(0., 3., N)
X = odeint(lambda x, t: np.dot(A+B*np.cos(t), x), x0, tt)


X_gps = [gpode.gaussianprocesses.GaussianProcess(
    gpode.kernels.Kernel.SquareExponKernel([1., v]))
         for v in [0.5, 1.1]]

for k, gp in enumerate(X_gps):
    gp.fit(tt, X[:, k])


def updf(a, r, k, j, As, G):
    As_ = [A.copy() for A in As]
    As_[r][k, j] = a

    F = np.sum([np.array([np.dot(A*g_, x) for g_, x in zip(g, X)])
                for A, g in zip(As_, G)], axis=0)

    L = X_gps[k].L
    mk = X_gps[k].pred(tt)
    etak = F[:, k] - mk

    return np.exp(-0.5*np.dot(etak,
                              np.linalg.solve(L.T,
                                              np.linalg.solve(L, etak))))


def updf2(a, r, k, j, As, G):
    As_ = [A.copy() for A in As]
    As_[r][k, j] = a

    F = np.sum([np.array([np.dot(A*g_, x) for g_, x in zip(g, X)])
                for A, g in zip(As_, G)], axis=0)

    exp_arg = 0.
    for k in range(K):
        L = X_gps[k].L
        mk = X_gps[k].pred(tt)
        etak = F[:, k] - mk
        exp_arg += -0.5*np.dot(etak,
                               np.linalg.solve(L.T, np.linalg.solve(L, etak)))

    return np.exp(exp_arg)


def arkj_par(r, k, j, As, G, X, mk, Ckinv):
    F = np.zeros(X.shape)
    for A, g in zip(As, G):
        F += np.array([np.dot(A*g_, x) for g_, x in zip(g, X)])

    mrkj = As[r][k, j]*X[:, j]*G[r] - F[:, k] + mk
    var = 1./np.dot(X[:, j], np.dot(Ckinv, X[:, j]))

    return var*np.dot(X[:, j]*G[r], np.dot(Ckinv, mrkj)), var


As = [A, B, C]
Gs = [np.ones(tt.size), np.cos(tt), np.sin(tt)]

r = 0
k = 1
j = 0

xj = X[:, j]
Lk = X_gps[k].L
Ckinv = np.linalg.inv(np.dot(Lk.T, Lk))

m0, v0 = arkj_par(r, k, j, As, Gs, X, X_gps[k].pred(tt), Ckinv)
l1 = m0 - 4*np.sqrt(v0)
l2 = m0 + 4*np.sqrt(v0)

n_const1, e1 = quad(lambda a: updf(a, r, k, j, As, Gs), l1, l2)
n_const2, e2 = quad(lambda a: updf2(a, r, k, j, As, Gs), l1, l2)

m1, e1 = quad(lambda a: a*updf(a, r, k, j, As, Gs)/n_const1, l1, l2)
m2, e2 = quad(lambda a: a*updf2(a, r, k, j, As, Gs)/n_const2, -np.inf, np.inf)

v1, e1 = quad(lambda a: (a-m1)**2*updf(a, r, k, j, As, Gs)/n_const1, l1, l2)
v2, e2 = quad(lambda a: (a-m2)**2*updf2(a, r, k, j, As, Gs)/n_const2, -np.inf, np.inf)

#print(m1, m2)
#print(v1, v2)


#print(1./np.dot(xj, np.dot(Ckinv, xj)))

R = 3
expr0 = As[r][k, j]*X[:, j]*Gs[r]
expr1 = np.sum([As[s][k, j]*X[:, j]*Gs[s] for s in range(R) if s != r], axis=0)
expr2 = np.sum([np.sum([As[r][k, l]*X[:, l]*Gs[r] for r in range(3)], axis=0)
                for l in range(K) if l != j], axis=0)

mrkj = X_gps[k].pred(tt) - (expr1 + expr2)
m3 = np.dot(xj, np.dot(Ckinv, mrkj))/np.dot(xj, np.dot(Ckinv, xj))
#print(m3)

#print(expr0 + expr1 + expr2)

F = np.array([np.dot(A+np.cos(t)*B+np.sin(t)*C, x)
              for t, x in zip(tt, X)])
mbar_rkj = As[r][k, j]*X[:, j] - F[:, k] + X_gps[k].pred(tt)

prec = np.dot(xj*Gs[r], np.dot(Ckinv, xj*Gs[r]))
#print(m1, m2, m3)
#print(np.dot(xj*Gs[r], np.dot(Ckinv, mbar_rkj)/prec))



A2 = [A.copy() for A in As]
A2[r][0, 0] = 4.
print(updf(0.52, r, k, j, As, Gs))
print(updf(0.52, r, k, j, A2, Gs))
#print(m0, v0)
#print(m1, v1)

#print(m2, v2)
"""

"""
k = 0

print(F[:, k])

r = 0
j = 1

expr0 = As[r][k, j]*X[:, j]*Gs[r]
expr1 = np.sum([As[s][k, j]*X[:, j]*Gs[s] for s in range(3) if s != r], axis=0)
expr2 = np.sum([np.sum([As[r][k, l]*X[:, l]*Gs[r] for r in range(3)], axis=0)
                for l in range(K) if l != j], axis=0)

print(expr0+expr1+expr2)
"""
