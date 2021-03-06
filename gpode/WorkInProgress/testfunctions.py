import numpy as np
from lindod_src import MGPLinearAdapGrad3
import matplotlib.pyplot as plt

def test1(mod):
    from lindod_src import xk_post_conditional
    from scipy.stats import multivariate_normal
    from scipy.optimize import minimize

    C0 = np.array([[0.6, 0.5, 0.1],
                   [0.0, 0.6, 0.3],
                   [0.0, 0.0, 1.0]])
    C0 = C0 + C0.T
    C0inv = np.linalg.inv(C0)

    cs = np.linspace(0.5, 1.5, 3)
    means = [np.random.normal(size=3) for k in range(3)]
    inv_covs = [C0inv/c for c in cs]

    def _lp(x):
        val = 0.
        for c, m in zip(cs, means):
            val += multivariate_normal.logpdf(x, mean=m, cov=C0*c)
        return -val

    res = minimize(_lp, np.zeros(3))
    print(res.x)
    print(res.hess_inv)

    m, C = prod_norm_pars(means, inv_covs)
    print(m)
    print(C)
    """
    i = 0
    means = []
    cov_invs = []
    for k in range(mod.output_dim):
        mk = mod._dmcs[k]
        Ck = mod._dCs[k] + np.diag(mod.gammas[k]*np.ones(mk.size))
        Lk = np.linalg.cholesky(Ck)

        m, Cinv = parse_comp_k(i, k, mod.latent_X, mod.As, mod.Gs, mk, Lk)
        means.append(m)
        cov_invs.append(Cinv)
    """


def test2(mod):
    # Tests the update step for _phi_k_mh_update
    from gpode.bayes import Parameter, ParameterCollection
    from lindod_src import MGPLinearAdapGrad3, _log_eq20_k, _x_gp_pars
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    K = 2
    xkp = []
    for k in range(K):
        p1 = Parameter("phi_{}_1".format(k),
                       prior=("gamma", (4, 0.2)),
                       proposal=("normal rw", 0.1))
        p2 = Parameter("phi_{}_2".format(k),
                       prior=("gamma", (4, 0.2)),
                       proposal=("normal rw", 0.1))
        phi_k = ParameterCollection([p1, p2], independent=True)
        xkp.append(phi_k)

    m = MGPLinearAdapGrad3(xkp)
    m.model_setup()
    m.data = mod.data
    m._X = mod.latent_X
    m._dXdt = mod._dXdt
    m._As = mod.As
    m._Gs = mod.Gs
    m._gammas = mod.gammas

    rv_phi_0 = []
    rv_phi_1 = []
    N_SIM = 2000
    BURN_IN = N_SIM/4
    for nt in range(N_SIM):
        for k in range(2):
            m._phi_k_mh_update(k)
        if nt > BURN_IN and nt % 10 == 0:
            rv_phi_0.append(m._x_kernels[0].kpar.value())
            rv_phi_1.append(m._x_kernels[1].kpar.value())

    rv_phi_0 = np.array(rv_phi_0)
    rv_phi_1 = np.array(rv_phi_1)

    fig1 = plt.figure()
    ax = fig1.add_subplot(121)
    ax.plot(rv_phi_0)
    ax = fig1.add_subplot(122)
    ax.plot(rv_phi_1)

    print(np.mean(rv_phi_0, axis=0))
    print(np.mean(rv_phi_1, axis=0))

    def objfunc(z):
        try:
            val = 0.
            zks = [z[:2], z[2:]]
            for k, zk in enumerate(zks):
                kern = m._x_kernels[k]
                kpar = kern.kpar

                xk = m._X[:, k]
                fk = m._dXdt(m._X, m._As, m._Gs)[:, k]

                Lxx, mdx_x, Cdxdx_x = _x_gp_pars(zk,
                                                 kern,
                                                 m.data.time,
                                                 m._X[:, k])

                val += _log_eq20_k(xk, fk, mdx_x,
                                   Lxx, Cdxdx_x,
                                   m._gammas[k],
                                   phi_k_val=zk, phi_k_prior=kpar.prior)
            return -val
        except:
            return np.inf

    res = minimize(objfunc, np.ones(4))
    print(res.x[:2])
    print(res.x[2:])

    plt.show()


def test3(mod):
    from lindod_src import _log_eq20, _log_eq20_k, _x_gp_pars
    from scipy.stats import norm
    from scipy.optimize import minimize
    m = mlfm_setup(mod)

    m._gammas = [0., 0.]
    m._sigmas = [1.3, 0.3]

    def f(x, i):
        _X = m._X.copy()
        _X[:, i] = x
        return _log_eq20(_X, m._As, m._Gs,
                         m._Lxxs, m._mdxs, m._dCds,
                         m._gammas, m._dXdt)

    def g(x, k):
        return np.sum(norm.logpdf(m.data.Y[:, k], loc=x, scale=m._sigmas[k]))

    def h(x, i, k):
        _X = m._X.copy()
        _X[:, i] = x
        F = m._dXdt(_X, m._As, m._Gs)

        Lxx, mdx, Cdxdx = _x_gp_pars(m._x_kernels[k].kpar.value(),
                                     m._x_kernels[k],
                                     m.data.time,
                                     _X[:, k])
        assert(np.all(Lxx == m.Lxx[k]))

        return _log_eq20_k(F[:, k], mdx,
                           Lxx, Cdxdx, m._gammas[k])

    def hsum(x, i):
        val = 0.
        for k in range(m.data.Y.shape[1]):
            val += h(x, i, k)
        return val

    _store_gpdx_covs(m)
    np.set_printoptions(precision=4)
    i = 1

    mhat, C = _parse_components(m, i)

    # mhatman
    mi0, cinv_i0 = _parse_component_k2(m, i, 0, ret_inv=True)
    print("------ k=0 ------")
    _res0 = minimize(lambda z: -h(z, i, 0), x0=m.data.Y[:, i])
    print(mi0)
    print(_res0.x)
    print("inv covs")
    print(cinv_i0)
    print(np.linalg.inv(_res0.hess_inv))
    print(-h(mi0, i, 0) < _res0.fun)
    print("------ k=1_------")
    mi1, cinv_i1 = _parse_component_k2(m, i, 1, ret_inv=True)
    _res1 = minimize(lambda z: -h(z, i, 1), x0=m.data.Y[:, i])
    print("means...")
    print(mi1)
    print(_res1.x)
    print("inv covs")
    print(cinv_i1)
    print(np.linalg.cond(cinv_i1))
    H = np.linalg.inv(_res1.hess_inv)
    print(H)
#    print(_res1.hess_inv)
    print(-h(mi1, i, 1) < _res1.fun)
    print("-----------")
    res = minimize(lambda z: -h(z, i, 0)-h(z, i, 1),
                   x0=m.data.Y[:, i], method="Nelder-Mead")
    
    A = cinv_i0 + cinv_i1
    y = np.dot(cinv_i0, mi0) + np.dot(cinv_i1, mi1)
    mhat2 = np.linalg.solve(A, y)
    print(res.x)
    print(mhat2)
    print(mhat)
    print("===============")

    print("res.fun(res.x)", res.fun)
    print("fun(mhat)", -hsum(mhat, i))
    print("fun(mhat2)", -h(mhat2, i, 0)-h(mhat2, i, 1))
    print(-h(mhat, i, 0)-h(mhat, i, 1) < res.fun)
    """

    k = 1
    xi = m._X[:, i]
#    mean, cov = parse_component_k(xi, i, k, m._x_kernels[k], m)
#    print(mean)

    _store_gpdx_covs(m)
    m2, c2 = _parse_component_k2(m, i, k)
    print(m2)
    print(c2)

    component_k(xi, i, k, m._x_kernels[k], m)
    h(xi, i, k)
    res = minimize(lambda z: -h(z, i, k),
                   x0=m.data.Y[:, i])

    print(res.x)
#    print(cov)
    print(res.hess_inv)
    print("--------")
    print(res.fun)
    print(-h(m2, i, k))
    print(-h(m2, i, k) < res.fun)
    """
    #m, C = parse_comp_k(i, k, m._X, m._As, m._Gs, m._mdxs[k], m._Lxxs[k])
    #print(m)
#    print(np.diag(res.hess_inv))

    """
    var = np.diag(res.hess_inv)
    sd = np.sqrt(var)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.fill_between(m.data.time, res.x - 2*sd, res.x+2*sd, alpha=0.2)
    ax.plot(m.data.time, res.x, '+')    
    ax.plot(m.data.time, m.data.Y[:, i], 's')
    plt.show()
    """
    """
    ms = []
    Cs = []
    for k in range(2):
        _, mdx_k, dCd_xk = _x_gp_pars(m._x_kernels[k].kpar.value(),
                                      m._x_kernels[k],
                                      m.data.time,
                                      m._X[:, k])
        m, C = parse_comp_k(i, k, m._X, m._As, m._Gs, ml, Lk)
    """

#    plt.plot(m.data.time, res.x, '+')
#    plt.plot(m.data.time, m.data.Y[:, k], 's')
#    plt.show()
def test4(mod):
    m = mlfm_setup(mod)
    m._gammas = [0.1, 0.1]
    m._sigmas = [0.5, 0.3]

def mlfm_setup(mod):
    from gpode.bayes import Parameter, ParameterCollection
    from lindod_src import _x_gp_pars
    K = 2
    xkp = []
    for k in range(K):
        p1 = Parameter("phi_{}_1".format(k),
                       prior=("gamma", (4, 0.2)),
                       proposal=("normal rw", 0.1))
        p2 = Parameter("phi_{}_2".format(k),
                       prior=("gamma", (4, 0.2)),
                       proposal=("normal rw", 0.1))
        phi_k = ParameterCollection([p1, p2], independent=True)
        xkp.append(phi_k)

    m = MGPLinearAdapGrad3(xkp)
    m.model_setup()
    m.data = mod.data
    m._X = mod.latent_X
    m._dXdt = mod._dXdt
    m._As = mod.As
    m._Gs = mod.Gs
    m._gammas = mod.gammas
    m._sigmas = mod.sigmas
    m._Lxxs = []
    m._mdxs = []
    m._dCds = []
    for i, kern in enumerate(m._x_kernels):
        Lxx, mdx_x, Cdxdx_x = _x_gp_pars(kern.kpar.value(), kern,
                                         m.data.time,
                                         m._X[:, i])
        m._Lxxs.append(Lxx)
        m._mdxs.append(mdx_x)
        m._dCds.append(Cdxdx_x)
    return m


def _store_gpdx_covs(m):
    m.Lxx = []
    m.Cxdx = []
    m.Cdxdx = []
    m.cLdxdx = []
    for k in range(m.data.Y.shape[1]):
        kern = m._x_kernels[k]
        tt = m.data.time
        Cxx = kern.cov(0, 0, tt, tt)
        m.Lxx.append(np.linalg.cholesky(Cxx))
        m.Cxdx.append(kern.cov(0, 1, tt, tt))

        Cdxdx = kern.cov(1, 1, tt, tt)
        m.Cdxdx.append(kern.cov(1, 1, tt, tt))

        Lxx = np.linalg.cholesky(Cxx)
        Cxdx = kern.cov(0, 1, tt, tt)
        cCdxdx = np.dot(Cxdx.T,
                        np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, Cxdx)))
        cCdxdx = Cdxdx - cCdxdx

        S = cCdxdx + np.diag(m._gammas[k]**2*np.ones(Cdxdx.shape[0]))
        LS = np.linalg.cholesky(S)

        m.cLdxdx.append(LS)


def _parse_components(mobj, i):
    ms = []
    inv_covs = []
    for k in range(mobj.data.Y.shape[1]):
        _m, _Cinv = _parse_component_k2(mobj, i, k, ret_inv=True)
        ms.append(_m)
        inv_covs.append(_Cinv)
    mean, covar = prod_norm_pars(ms, inv_covs)
    return mean, covar


def _parse_component_k2(m, i, k, ret_inv=False):

    Lxx = m.Lxx[k]
    Cxdx = m.Cxdx[k]
    cLdxdx = m.cLdxdx[k]

    vv = []
    for j in range(m.data.Y.shape[1]):
        vj = np.sum([m._As[r][k, j]*g for r, g in enumerate(m._Gs)], axis=0)
        vv.append(vj)

    if i != k:
        xk = m._X[:, k]

        da = np.diag(vv[i])
        mk = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, xk)))

        b = mk - np.sum([v*m._X[:, j] for j, v in enumerate(vv)
                         if j != i], axis=0)

        cov_inv = np.dot(da,
                         np.linalg.solve(cLdxdx.T,
                                         np.linalg.solve(cLdxdx, da)))
        mean = np.dot(np.linalg.pinv(da), b)
        if ret_inv:
            return mean, cov_inv
        else:
            return mean, np.linalg.inv(cov_inv)

    else:
        I = np.diag(np.ones(Lxx.shape[0]))
        T = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, I)))

        b = - np.sum([v*m._X[:, j] for j, v in enumerate(vv)
                      if j != i], axis=0)

        A = np.diag(vv[i]) - T

        M1 = np.dot(A.T, np.linalg.solve(cLdxdx.T, np.linalg.solve(cLdxdx, A)))
        w1 = np.dot(b, np.linalg.solve(cLdxdx.T, np.linalg.solve(cLdxdx, A)))

        mean = np.linalg.solve(M1, w1)
        cov_inv = np.dot(A.T,
                         np.linalg.solve(cLdxdx.T, np.linalg.solve(cLdxdx, A)))
        cCdxdx = np.dot(cLdxdx.T, cLdxdx)
        mat = np.dot(A.T, np.dot(np.linalg.inv(cCdxdx), A))
#        assert(cCdxdx == m._x_kernels[k](1, 1
#        print(cLdxdx)
#        print(cov_inv)
#        print(mat)
#        assert(np.all(cov_inv == mat))
        if ret_inv:
            return mean, cov_inv
        else:
            return mean, np.linalg.pinv(cov_inv)


def parse_component_k(x, i, k, kern, m):
    tt = m.data.time
    Cxx = kern.cov(0, 0, tt, tt)
    Lxx = np.linalg.cholesky(Cxx)
    Cxdx = kern.cov(0, 1, tt, tt)
    Cdxdx = kern.cov(1, 1, tt, tt)

    Cdxdx_x = Cdxdx - np.dot(Cxdx.T,
                             np.linalg.solve(Lxx.T,
                                             np.linalg.solve(Lxx, Cxdx)))

    S = Cdxdx_x + np.diag(m._gammas[k]**2*np.ones(tt.size))
    Sinv = np.linalg.inv(S)
    L = np.linalg.cholesky(S)

    _X = m._X.copy()

    vv = []
    for j in range(m.data.Y.shape[1]):
        vj = np.sum([m._As[r][k, j]*g for r, g in enumerate(m._Gs)], axis=0)
        vv.append(vj)

    if i != k:
        a = vv[i]
        da = np.diag(a)

        mk = np.dot(Cxdx.T,
                    np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, _X[:, k])))

        _X[:, i] = x
        b = mk - np.sum([v*_X[:, j] for j, v in enumerate(vv)
                         if j != i], axis=0)

#        fk = m._dXdt(_X, m._As, m._Gs)[:, k]

        cov_inv = np.dot(da, np.linalg.solve(L.T, np.linalg.solve(L, da)))
        mean = np.dot(np.linalg.pinv(da), b)
        print(np.linalg.solve(np.dot(da, np.dot(Sinv, da)), np.dot(da, np.dot(Sinv, b))))
        return mean, np.linalg.inv(cov_inv)

    else:
        I = np.diag(np.ones(tt.size))
        T = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, I)))

        mk = np.dot(Cxdx.T,
                    np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, _X[:, k])))
        b = - np.sum([v*_X[:, j] for j, v in enumerate(vv)
                      if j != i], axis=0)
        _X[:, i] = x
        fk = m._dXdt(_X, m._As, m._Gs)[:, k]

        A = np.diag(vv[i]) - T
        Sxxinv = np.dot(A.T, np.dot(Sinv, A))

        M1 = np.dot(A.T, np.dot(Sinv, A))
        w1 = np.dot(b, np.dot(Sinv, A))
        w2 = np.dot(A.T, np.dot(Sinv, b))
#        print("========")
#        print(np.dot(A, x) - b)        
#        print(w1)
#        print(w2)
#        print("========")        
#        print(np.linalg.solve(M1, w1))
#        print(np.dot(A, x))
#        print(vv[i]*x - mk)
#        print(A, np.linalg.inv(A))
        mean = np.linalg.solve(M1, w1)#np.dot(np.linalg.pinv(A), b)
#        print(fk - mk)
#        print(np.dot(A, x) - b)

        return mean, np.linalg.inv(np.dot(A.T, np.dot(Sinv, A)))

def component_k(x, i, k, kern, m):
    tt = m.data.time
    Cxx = kern.cov(0, 0, tt, tt)
    Lxx = np.linalg.cholesky(Cxx)
    Cxdx = kern.cov(0, 1, tt, tt)
    Cdxdx = kern.cov(1, 1, tt, tt)

    _X = m._X.copy()
    _X[:, i] = x

    fk = m._dXdt(m._X, m._As, m._Gs)[:, k]
    mk = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, _X[:, k])))

    Cdxdx_x = Cdxdx - np.dot(Cxdx.T,
                             np.linalg.solve(Lxx.T,
                                             np.linalg.solve(Lxx, Cxdx)))

    S = Cdxdx_x + np.diag(m._gammas[k]**2*np.ones(tt.size))
    L = np.linalg.cholesky(S)

    return norm_quad_form(fk - mk, L)


def norm_quad_form(x, L):
    return -0.5*np.dot(x, np.linalg.solve(L.T, np.linalg.solve(L, x)))

#########
#
# parse the expression exp(-0.5*(fk - mk)K^{-1}(fk-mk))
#
# returns: a, b where a, b are the components of a
#
#     a ⊙ x_i - b = fk - mk
#
##########################
def parse_comp_k(i, k, X, As, Gs, mk, L):

    a = np.sum([As[r][k, i]*g for r, g in enumerate(Gs)], axis=0)
#    print(a)
#    print(As[0][k, i]*Gs[0] + As[1][k, i]*Gs[1])
    b = [X[:, j]*np.sum([As[r][k, j]*g for r, g in enumerate(Gs)], axis=0)
         for j in range(X.shape[1]) if j != i]
    b = np.sum(b, axis=0)

    if np.all(a == 0):
        mean = np.zeros(a.size)
    else:
        mean = np.linalg.solve(np.diag(a), b)
    cov_inv = np.dot(np.diag(a),
                 np.linalg.solve(L.T, np.linalg.solve(L, np.diag(a))))

    return mean, cov_inv


def parse_comp_k2(i, k, X, As, Gs, mk,
                  Lxx, Cxdx, Cdxdx):
    vv = []
    for j in range(X.shape[1]):
        vj = np.sum([As[r][k, j]*g for r, g in enumerate(m._Gs)], axis=0)
        vv.append(vj)

    if i == k:
        mean = 0
        cov = 0
    else:
        mean = 0
        cov = 0

##################################################################
# Returns the parameter of the normal distribution given by a    #
# product of the densities                                       #
#          ___                                                   #
#   p(x) ∝ | | N(x | means[k], inv_covs[k]^{-1})                 #
#           k                                                    #
##################################################################
def prod_norm_pars(means, inv_covs):

    m1 = means[0]
    C1inv = inv_covs[0]
    for m2, C2inv in zip(means[1:], inv_covs[1:]):
        Cnew_inv = C1inv + C2inv
        Cnew = np.linalg.inv(Cnew_inv)
        mnew = np.dot(Cnew, np.dot(C1inv, m1) + np.dot(C2inv, m2))

        C1inv = Cnew_inv
        m1 = mnew

    return mnew, Cnew
