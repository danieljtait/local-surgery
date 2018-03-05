import numpy as np
from scipy.integrate import odeint
from scipy.stats import multivariate_normal
np.set_printoptions(precision=2)
np.random.seed(11)

mu = 2.

Zero = np.zeros((2, 2))
L00 = np.array([[mu, 0],
                [1/mu, 0]])
L12 = np.random.normal(size=4).reshape(2, 2)
L13 = np.random.normal(size=4).reshape(2, 2)
L21 = np.random.normal(size=4).reshape(2, 2)
#np.array([[-mu, 0.],
#                [0., 0.]])
#L13 = np.array([[1., 0.],
#                [0., -0.5]])
#L21 = np.array([[0., -mu],
#                [0., 0.]])


Model_L = [[L00],
           [Zero, L12, L13],
           [L21]]

from collections import namedtuple

Moments = namedtuple("Moments", "mean cov")


class PolyVarMLFM:

    def __init__(self,
                 ModelMatrices,
                 data_time =None, data_Y = None):

        # handle model structure
        self.ModelMatrices = np.asarray([m_ls for m_ls in ModelMatrices])
        self._degrees = [len(m_ls) for m_ls in ModelMatrices]

        self._R = len(self.ModelMatrices)-1
        self._N = 3

        self._init_G_var_dist()
        
    def _init_G_var_dist(self):

        self._aux_lf = {}
        for r, Dr in enumerate(self._degrees):
            _aux_lf_r = []
            for d in range(Dr-1):
                m0 = np.random.normal(size=self._N)
                _m = Moments(m0, np.diag(np.ones(self._N)))
                _aux_lf_r.append(_m)
            self._aux_lf[r] = _aux_lf_r

            


"""
Model Fit Utility Functions
"""


def _prod(x_list):
    res = x_list[0]
    for x in x_list[1:]:
        res *= x
    return res

####
# Matrix RV util function 
#
# returns the covariance matrix of
#
# P1 = x1 circ ... circ xn
# P2 = x1 circ ... circ xn circ ... circ xn+p
#
###
def _ind_hadamard_prod_cov(n1, n2, EX, CovX):

    prod_ExixiT = np.ones((EX[0].size, EX[0].size))
    prod_mximxiT = np.ones((EX[0].size, EX[0].size))

    nmin = min(n1, n2)

    for i in range(nmin):
        mmT = np.outer(EX[i], EX[i])
        
        prod_ExixiT *= (CovX[i] + mmT)
        prod_mximxiT *= mmT

    CovPnPn = prod_ExixiT - prod_mximxiT

    if n1 == n2:
        return CovPnPn
    elif n1 < n2:
        return np.dot(CovPnPn, np.diag(_prod(EX[n1:n2])))
    elif n2 < n1:
        return np.dot(np.diag(_prod(EX[n2:n1])), CovPnPn)

##
# for the product g_{r2} ... g_{rDr} of independent
# gaussian processes return the covariance of the
# products of random variables
#
def _get_cov_prod_gp(degr, aux_lf_Er, aux_lf_Covr):
    cov_prod_gp = np.row_stack((
    np.column_stack((_ind_hadamard_prod_cov(d1+1, d2+1, aux_lf_Er, aux_lf_Covr))
        for d2 in range(degr-1)) for d1 in range(degr-1)))
    return cov_prod_gp

def _get_vi_rj_mean(i, j, Model_L_r, aux_lf_r, N):
    res = Model_L_r[0][i, j]*np.ones(N)
    Egprod = 1
    for d, lf in enumerate(aux_lf_r):
        Egprod *= lf.mean
        res += Model_L_r[d+1][i, j]*Egprod
    return res


def _get_vi_rj_vi_rl_cov(i, j, l, Model_L_r, cov_prod_gp, N):
    I = np.diag(np.ones(N))
    Lr1_vec = np.row_stack((Lrd[i, j]*I for Lrd in Model_L_r[1:]))
    Lr2_vec = np.row_stack((Lrd[i, l]*I for Lrd in Model_L_r[1:]))
    return np.dot(Lr1_vec.T, np.dot(cov_prod_gp, Lr2_vec))


def _get_vi_rjxj_vi_slxl_cov(i, r, s, j, l,
                             Model_L,
                             aux_lf,
                             EX, CovX,
                             N):

    if s != r:
        Evi_rj = _get_vi_rj_mean(i, j, Model_L[r], aux_lf[r], N)
        Evi_sl = _get_vi_rj_mean(i, l, Model_L[s], aux_lf[s], N)

        # Using the independence of for s != r of vi_rj and vi_sl
        Evi_rj_vi_slT = np.outer(Evi_rj, Evi_sl)

        Exj = EX[j]
        Exl = EX[l]
        ExjExlT = np.outer(Exj, Exl)
        ExjxlT = CovX[(j, l)] + ExjExlT

        return Evi_rj_vi_slT*ExjxlT - np.outer(Exj*Evi_rj, Exl*Evi_sl)

    else:
        degr = len(Model_L[r])
        if degr == 1:
            return np.zeros((N, N))

        aux_lf_r_E = [lf.mean for lf in aux_lf[r]]
        aux_lf_r_Cov = [lf.cov for lf in aux_lf[r]]
        
        cov_prod_gp = _get_cov_prod_gp(degr, aux_lf_r_E, aux_lf_r_Cov)
        cov_vi_rj_vi_rl = _get_vi_rj_vi_rl_cov(i, j, l, Model_L[r],
                                               cov_prod_gp, N)

        Evi_rj = _get_vi_rj_mean(i, j, Model_L[r], aux_lf[r], N)
        Evi_rl = _get_vi_rj_mean(i, l, Model_L[r], aux_lf[r], N)

        Evi_rj_vi_rlT = cov_vi_rj_vi_rl + np.outer(Evi_rj, Evi_rl)

        Exj = EX[j]
        Exl = EX[l]
        ExjExlT = np.outer(Exj, Exl)

        ExjxlT = CovX[(j, l)] + ExjExlT

        return Evi_rj_vi_rlT*ExjxlT - np.outer(Exj*Evi_rj, Exl*Evi_rl)


def _get_Vi_mean(i, EX, aux_lf, Model_L, R, K, N):
    EVi = []
    for r in range(R+1):
        Evrj = [_get_vi_rj_mean(i, j, Model_L[r], aux_lf[r], N)
                for j in range(K)]
        EVi.append( sum(evrj*ex for evrj, ex in zip(Evrj, EX)) )
    return EVi

obj = PolyVarMLFM(Model_L)

#print(obj.ModelMatrices[1])
#print(obj._aux_lf[1])

EX = [np.random.normal(size=3), np.random.normal(size=3)]
K = 2
R = 2

#print(_get_Vi_mean(1, EX, obj._aux_lf, obj.ModelMatrices, R, K, obj._N))
tt = np.linspace(0., 2, 6)
_cov = np.array([[np.exp(-(s-t)**2) for t in tt] for s in tt])
EX = [np.cos(tt[:3]), np.sin(tt[3:])]
CovX = {}
CovX[(0, 0)] = _cov[:3, :3]
CovX[(0, 1)] = _cov[:3, 3:]
CovX[(1, 0)] = _cov[3:, :3]
CovX[(1, 1)] = _cov[3:, 3:]

i = 0
j = 1
l = np.random.choice([0, 1])

for r in range(obj._R+1):
    for s in range(r+1):
        print(r,s)
        c = _get_vi_rjxj_vi_slxl_cov(i, r, s, j, l,
                                     obj.ModelMatrices,
                                     obj._aux_lf,
                                     EX, CovX,
                                     obj._N)
        print(c)
"""
def _get_vi_rj(i, j, Model_L_r, aux_g, N):
    res = Model_L_r[0][i, j]*np.ones(N)
    gprod = 1
    for d, g in enumerate(aux_g):
        gprod *= g
        res += Model_L_r[d+1][i, j]*gprod
    return res

for r, degr in enumerate(obj._degrees):

    if degr > 1:

        aux_lf_r_E = [lf.mean for lf in obj._aux_lf[r]]
        aux_lf_r_Cov = [lf.cov for lf in obj._aux_lf[r]]    


        nsim = 10

        val1 = np.zeros(3)
        val2 = np.zeros(3)
        EvvT = np.zeros((3, 3))

        i = 1
        j = 1
        l = 1

        Evi_rjxj = np.zeros(3)
        Evi_rlxl = np.zeros(3)
        nsim = 50000

        val2 = np.zeros((3, 3))

        for nt in range(nsim):
            x1x2 = multivariate_normal.rvs(np.concatenate(EX), _cov)
            xj = x1x2[j*3:j*3+3]
            xl = x1x2[l*3:l*3+3]
            
            gr2 = multivariate_normal.rvs(aux_lf_r_E[0], aux_lf_r_Cov[0])
            gr3 = multivariate_normal.rvs(aux_lf_r_E[1], aux_lf_r_Cov[1])

            vi_rjxj = xj*_get_vi_rj(i, j, Model_L[r], [gr2, gr3], 3)
            vi_rlxl = xl*_get_vi_rj(i, l, Model_L[r], [gr2, gr3], 3)

            Evi_rjxj += vi_rjxj
            Evi_rlxl += vi_rlxl

            val2 += np.outer(xj, xl)
            
            EvvT += np.outer(vi_rjxj, vi_rlxl)

        EvvT /= nsim
        Evi_rjxj /= nsim
        Evi_rlxl /= nsim
        val2 /= nsim

        print(EvvT - np.outer(Evi_rjxj, Evi_rlxl))
        print(_get_vi_rjxj_vi_slxl_cov(i, r, r, j, l, Model_L, obj._aux_lf,
                                       EX, CovX, 3))
"""
