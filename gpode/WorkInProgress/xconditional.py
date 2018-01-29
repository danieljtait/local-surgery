import numpy as np


###
#
# We assume that somewhere mobj has
def xi_conditional(mobj, i):
    pass


"""
Hi
"""


##
# Attaches the GP kernel matrices to the model
def _store_gpdx_covs(mobj):
    mobj.Lxx = []
    mobj.Cxdx = []
    mobj.cLdxdx = []

    for k in range(mobj.data.Y.shape[1]):

        kern = mobj._x_kernels[k]
        tt = mobj.data.time

        Cxx = kern.cov(0, 0, tt, tt)
        Lxx = np.linalg.cholesky(Cxx)
        Cxdx = kern.cov(0, 1, tt, tt)
        Cdxdx = kern.cov(1, 1, tt, tt)

        cCdxdx = np.dot(Cxdx.T,
                        np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, Cxdx)))
        cCdxdx = Cdxdx - cCdxdx

        L = np.linalg.cholesky(cCdxdx)

        mobj.Lxx.append(Lxx)
        mobj.Cxdx.append(Cxdx)
        mobj.cLdxdx.append(L)


def _parse_component_k(mobj, i, k, ret_inv=False):

    Lxx = mobj.Lxx[k]
    Cxdx = mobj.Cxdx[k]
    cLdxdx = mobj.cLdxdx[k]

    vv = []
    for j in range(mobj.data.Y.shape[1]):
        vj = np.sum([mobj._As[r][k, j]*g
                     for r, g in enumerate(mobj._Gs)], axis=0)
        vv.append(vj)

    if i != k:
        xk = mobj._X[:, k]

        A = np.diag(vv[i])
        mk = np.dot(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, xk)))

        b = mk - np.sum([v*mobj._X[:, j] for j, v in enumerate(vv)
                         if j != i], axis=0)

    else:
        I = np.diag(np.ones(Lxx.shape[0]))
        T = np.diag(Cxdx.T, np.linalg.solve(Lxx.T, np.linalg.solve(Lxx, I)))

        b = -np.sum([v*mobj._X[:, j] for j, v in enumerate(vv)
                     if j != i], axis=0)

        A = np.diag(vv[i]) - T

    cov_inv = np.dot(A.T,
                     np.linalg.solve(cLdxdx.T,
                                     np.linalg.solve(cLdxdx, A)))

    if ret_inv:
        return mean, cov_inv
