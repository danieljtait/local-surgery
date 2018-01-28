import numpy as np


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
    ### Tests the update step for _phi_k_mh_update
    from gpode.bayes import Parameter, ParameterCollection
    from lindod_src import MGPLinearAdapGrad3

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
    m._X = mod.latent_X
    m._phi_k_mh_update(0)

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
