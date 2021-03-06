import numpy as np
import collections

Moments = collections.namedtuple("Moments", "mean cov")


class PolyVarMLFM:

    def __init__(self,
                 ModelMatrices,
                 data_time = None, data_Y = None,
                 sigmas = None,
                 gammas = None,
                 x_gp_pars = None):

        # handle model structure
        self.ModelMatrices_offset = ModelMatrices[0]
        self.ModelMatrices = [m_ls for m_ls in ModelMatrices[1:]]
        self._degrees = [len(m_ls) for m_ls in ModelMatrices[1:]]

        self._R = len(ModelMatrices) - 1

        # Attach data and check size
        self.data_time = data_time
        self.data_Y = data_Y

    def _init_G_var_dist(self):

        _principal_means = {}
        _principal_covars = {}

        _auxillary_means = {}
        _auxillary_covars = {}
        
        for r, deg in enumerate(self._degrees):
            _principal_means[r] = np.zeros(self._N)
            for s in range(r+1):
                _principal_covars[(r, s)] = np.diag(np.ones(self._N))
                _principal_covars[(s, r)] = np.diag(np.ones(self._N))

            for d in range(1, deg):
                _auxillary_means[(r, d)] = np.zeros(self._N)
                _axxillary_covars[(r, d)] = np.diag(np.ones(self._N))

            
######################################################
#                                                    #
# vi_rj = ( L_{ij}^{r,1}                             #
#            + sum_{d=2}^{Dr}L_{ij}^{r,d}            #
#             x prod_{p=}2^{d} g_rp )                #
#                                                    #
######################################################
def _get_vi_rj_mean(i, r, j, degr, Lr, aux_lf_Er, aux_lf_Cov):
    result = Lr[0, i, j]
    Egprod = 1
    for d in range(1, degr):
        Egprod *= aux_lf_Er[d-1]        
        result += Lr[d, i, j]*Egprod
    return result

def _get_vi_rj_cov(i, r, j, k, degr, Lr, aux_lf_Er, aux_lf_Covr, N):
    cov_prod_gp = np.row_stack((
        np.column_stack((_ind_hadamard_prod_cov(d1+1, d2+1, aux_lf_Er, aux_lf_Covr))
                        for d2 in range(degr-1))
        for d1 in range(degr-1)))

    I = np.diag(np.ones(N))
    Lr1_vec = np.row_stack((Lrd1[i, j]*I for Lrd1 in Lr[1:]))
    Lr2_vec = np.row_stack((Lrd2[i, k]*I for Lrd2 in Lr[1:]))

    res = np.dot(Lr1_vec.T, np.dot(cov_prod_gp, Lr2_vec))
    return res

def _get_vr_vs_cov(i, r, s, Lr, Ls,
                   degr, degs,
                   aux_lf_Er, aux_lf_Covr,
                   EX_full, EXXT,
                   K, N,
                   aux_lf_Es=None, aux_lf_Covs=None):
    
    if r == s:

        Evv = np.concatenate([_get_vi_rj_mean(i, None, j, degr, Lr, aux_lf_Er, aux_lf_Covr)
                              for j in range(K)])
        Covvv = np.row_stack(
            (np.column_stack((_get_vi_rj_cov(i, None, j, k, degr, Lr, aux_lf_Er, aux_lf_Covr, N)
                              for k in range(K)))
             for j in range(K))
             )
        EvvT = Covvv + np.outer(Evv,Evv)
        Cov_vx = EvvT * EXXT - np.outer(Evv*EX_full, Evv*EX_full)

        Ivec = np.row_stack((np.diag(np.ones(N)) for k in range(K)))

        return np.dot(Ivec.T, np.dot(Cov_vx, Ivec))

    else:
        Evvr = np.concatenate([_get_vi_rj_mean(i, None, j, degr, Lr, aux_lf_Er, aux_lf_Covr)
                               for j in range(K)])

        Evvs = np.concatenate([_get_vi_rj_mean(i, None, j, degs, Ls, aux_lf_Es, aux_lf_Covs)
                               for j in range(K)])

        EvrvsT = np.outer(Evvr, Evvs)  # By independence

        Cov_vx = EvrvsT * EXXT - np.outer(Evvr*EX_full, Evvs*EX_full)

        Ivec = np.row_stack((np.diag(np.ones(N)) for k in range(K)))
        return np.dot(Ivec.T, np.dot(Cov_vx, Ivec))
        

####
def _get_Vi_cov(i):
   
    EX_full = np.concatenate([ex for ex in EX])
    CovX_full = np.row_stack(
        (
        np.column_stack((CovX[(j, k)] for k in range(K))
        )
         for j in range(K)))

    EXXT = CovX_full + np.outer(EX_full, EX_full)

    res_dict = {}

    for r in range(R):
        for s in range(r+1):
            cov_vr_vs = _get_vr_vs_cov(i, r, s,
                                       L[r], L[s],
                                       aux_lf[r].mean, aux_lf[r].cov,
                                       aux_lf[s].mean, aux_lf[s].cov,
                                       EX_full, EXXT,
                                       K, N)
    

######################################################
#
# Vi is the block matrix with components diag(vr)
# where
#
#     vr = sum_j v_{rj} circ x_j
#
# with
#
#     vrj = L_{ij}^{(r, 1)}
#           + sum_{d=2}^{Dr}L_{ij}^{(r, d)}
#
######################################################

def _get_vi_rj_mean2(i, j, Model_L_r, aux_lf_r):
    res = Model_L_r[0, i, j]
    Egprod = 1
    for d, lf in enumerate(aux_lf_r):
        Egprod *= lf.mean
        res += Lr[d+1, i, j]*Egprod
    return res

def _get_vi_rj_cov2(i, j, Model_L_r, cov_prod_gp):
    
    cov_prod_gp = np.row_stack((
        np.column_stack((_ind_hadamard_prod_cov(d1+1, d2+1,
                                                aux_lf_means, aux_lf_covs)
                         for d2 in range(degr-1)))
        for d1 in range(degr-1)))

    I = np.diag(np.ones(aux_lf_means[0].size))
    Lr1_vec = np.row_stack((Lrd[i, j]*I for Lrd in Model_L_r[1:]))
    Lr2_vec = np.row_stack((Lrd[i, k]*I for Lrd in Model_L_r[1:]))

    return np.dot(Lr1_vec.T, np.dot(cov_prod_gp, Lr2_vec))

def _get_Vi_mean(i, EX, aux_lf, Model_L, R):
    EVi = []
    for r in range(R+1):
        Evrj = [_get_vi_rj_mean2(i, r, j, Model_L[r], aux_lf[r])
                for j in range(k)]
        EVi.append( sum(evrj*ex for evrj, ex in zip(Evrj, EX)) )
    return EVi
    
def _parse_component_i_for_gprincipal(i,
                                      EX, CovX,
                                      aux_lf,
                                      Model_L):
    E_Vi = _get_Vi_mean(i, EX, aux_lf, Model_L)
        
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
    
        
        

from scipy.stats import multivariate_normal
def _sim_vi_rj(i, r, j, degr, Lr, gs):


    result = Lr[0, i, j]
    gprod = 1
    for d in range(1, degr):
        gprod *= gs[d-1]
        result += Lr[d, i, j]*gprod

    return result


np.set_printoptions(precision=2)
np.random.seed(11)

ttx = np.linspace(0., 1., 6)
covX = np.array([[np.exp(-(s-t)**2) for t in ttx] for s in ttx])
mx = [np.cos(ttx[:3]), np.sin(ttx[3:])]
covX_dict = {}
covX_dict[(0, 0)] = covX[:3, :3]
covX_dict[(0, 1)] = covX[:3, 3:]
covX_dict[(1, 0)] = covX[3:, :3]
covX_dict[(1, 1)] = covX[3:, 3:]

tt = np.linspace(0., 1., 3)
cov = np.array([[np.exp(-(s-t)**2) for t in tt] for s in tt])

degr = 3

aux_lf_Er = [np.cos(tt), np.cos(tt) + 0.1*tt**2, np.cos(tt) - 0.1*tt]
aux_lf_Covr = [cov, 1.3*cov, .7*cov]

Lr1 = np.random.normal(size=4).reshape(2, 2)
Lr2 = np.random.normal(size=4).reshape(2, 2)
Lr3 = np.random.normal(size=4).reshape(2, 2)
Lr = np.array([Lr1, Lr2, Lr3])

_get_vr_vs_cov(0, 1, 1, Lr, Lr,
               degr, degr,
               aux_lf_Er, aux_lf_Covr,
               mx, covX_dict,
               2, tt.size)
