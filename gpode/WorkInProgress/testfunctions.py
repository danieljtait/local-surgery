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
    from lindod_src import _log_eq20
    from scipy.stats import norm
    from scipy.optimize import minimize
    m = mlfm_setup(mod)

    m._sigmas = [0.1, 0.1]

    def f(x, k):
        _X = m._X.copy()
        _X[:, k] = x
        return _log_eq20(_X, m._As, m._Gs,
                         m._Lxxs, m._mdxs, m._dCds,
                         m._gammas, m._dXdt)

    def g(x, k):
        return np.sum(norm.logpdf(m.data.Y[:, k], loc=x, scale=m._sigmas[k]))

    k = 0
    res = minimize(lambda z: -f(z, k),
                   x0=m.data.Y[:, k])
    print(res)

#    plt.plot(m.data.time, res.x, '+')
#    plt.plot(m.data.time, m.data.Y[:, k], 's')
#    plt.show()

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

#########
#
# parse the expression exp(-0.5*(fk - mk)K^{-1}(fk-mk))
#
# returns: a, b where a, b are the components of a
#
#     a ⊙ x - b = fk - mk
#
##########################
def parse_comp_k(i, k, X, As, Gs, mk, L):

    a = np.sum([As[r][k, i]*g for r, g in enumerate(Gs)], axis=0)
#    print(a)
#    print(As[0][k, i]*Gs[0] + As[1][k, i]*Gs[1])
    b = [X[:, j]*np.sum([As[r][k, j]*g for r, g in enumerate(Gs)], axis=0)
         for j in range(X.shape[1]) if j != i]
    b = np.sum(b, axis=0)
    b = mk - b

    if np.all(a == 0):
        mean = np.zeros(a.size)
    else:
        mean = np.linalg.solve(np.diag(a), b)
    cov_inv = np.dot(np.diag(a),
                 np.linalg.solve(L.T, np.linalg.solve(L, np.diag(a))))

    return mean, cov


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
