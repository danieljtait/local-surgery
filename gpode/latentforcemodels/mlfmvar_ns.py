import numpy as np
from collections import namedtuple
from .nestedgpintegrals import ngpintegrals_sqexpcovs
from scipy.stats import norm
from scipy.linalg import block_diag
from scipy.special import erf
import scipy.linalg
from itertools import chain

PsiMoments = namedtuple("PsiMoments", "mean cov")
ObsHyperPars = namedtuple("ObsHyperPars", "alpha beta")

### Rewrite!
class VarMLFM:
    def __init__(self, t0_ind, data_times, data_Y,
                 step_size,
                 As,
                 lscales=None,
                 obs_noise_priors=None):

        # Attach the covar. funcs
        self._cov_funcs = _cov_funcs

        # Models dimenstions
        As = np.asarray(As)                # Arrayify as
        self.A = As
        assert(As.shape[1] == As.shape[2]) # Squareness check 

        self._R = As.shape[0] - 1 # Total number of latent forces
        self._K = As[1]           # Dimensions of observations


        # Hyperprior set up for the observation noise scale
        self._obs_noise_priors = ObsHyperPars(obs_noise_priors[0],
                                              obs_noise_priors[1])

        # Hyperprior set up for inverse char. length scales of the
        # latent GPs
        self.lscales = lscales

        # Handle the seting up into times before and after t0_ind
        _setup_times(self, t0_ind, data_times, data_Y, step_size)

        # initalise the variationional distributions
        self._init_psi_distributions()

    def _update_psi_i(self, i, direc):
        _get_prior_cond_pars(i, self)

    def _func(self, i):
        return _get_prior_cond_pars(i, self)

    def _init_psi_distributions(self):
        _m0 = np.zeros(2*self._R)
        _I0 = np.diag(np.ones(2*self._R))

        bmean = np.array([_m0 for n in range(self._N_b_psi)])
        bcovar = np.array([_I0 for n in range(self._N_b_psi)])

        fmean = np.array([_m0 for n in range(self._N_f_psi)])
        fcovar = np.array([_I0 for n in range(self._N_f_psi)])

        self._backward_psi = PsiMoments(bmean, bcovar)
        self._forward_psi = PsiMoments(fmean, fcovar)

        # Also use this opporutinity to initalise the corresponding
        # moments to each of the variables Si so we don't do this on the
        # fly
        self._b_Si_moments = [[] for i in range(self._N_b_psi)]
        self._f_Si_moments = [[] for i in range(self._N_f_psi)]

        for n, direc in zip([self._N_b_psi, self._N_f_psi],
                            ["backward", "forward"]):
            for i in range(n):
                self._store_Si_moments(i, direc)


    def _store_Si_moments(self, i, direc):
        if direc == "backward":
            psi_i_mean = self._backward_psi.mean[i][:self._R]
            psi_i_cov = self._backward_psi.cov[i][:self._R, :self._R]
            dt = self.backward_dts[i]
            self._b_Si_moments[i] = _get_vec_Si_pars(psi_i_mean,
                                                     psi_i_cov,
                                                     self.A, dt)
        if direc == "forward":
            psi_i_mean = self._forward_psi.mean[i][:self._R]
            psi_i_cov = self._forward_psi.cov[i][:self._R, :self._R]
            dt = self.forward_dts[i]
            self._f_Si_moments[i] = _get_vec_Si_pars(psi_i_mean,
                                                     psi_i_cov,
                                                     self.A, dt)                
        
class VariationalMLFM2:

    def __init__(self,
                 t0_ind, data_times, data_Y,
                 step_size,
                 As=None,
                 lscales=None,
                 obs_noise_priors=None):

        self.A = np.asarray(As)
        self._R = len(As)-1
        self._K = self.A.shape[1]
        self.lscales = lscales

        # Initalise the priors for the scale
        self._obs_noise_priors = np.asarray(obs_noise_priors)
        self.obs_noise_pars = [np.array([3, 3]),
                               np.array([0.5, 0.5])]

        _setup_times(self, t0_ind, data_times, data_Y, step_size)
        _handle_cov_setup(self)        

        # Initalise the trajectory variational parameters
        self._init_psi_distributions()

        _init_ex_exxt(self,
                      self.forward_data[0],
                      np.outer(self.forward_data[0], self.forward_data[0]))



    def _update_noise_pars(self):

        a0 = self._obs_noise_priors[:, 0]
        b0 = self._obs_noise_priors[:, 1]
        
        N = len(self.forward_data_inds) + len(self.backward_data_inds) - 2

        alphas = a0 + N/2

        betas = np.zeros(self._K)
        for ni in self.forward_data_inds[1:]:
            yn = self._f_y_ind_map[ni]
            Exi = self._f_Ex[ni]
            Exxt = self._f_Exxt[ni]

            betas += yn**2 - 2*yn*Exi + np.diag(Exxt)

        for ni in self.backward_data_inds[1:]:
            yn = self._b_y_ind_map[ni]
            Exi = self._b_Ex[ni]
            Exxt = self._b_Exxt[ni]

            betas += yn**2 - 2*yn*Exi + np.diag(Exxt)

        betas *= 0.5
        betas += b0

        self.obs_noise_pars = [np.array(alphas),
                               np.array(betas)]
            
    def test(self):
        Dinv = np.diag([5., 5.])
        
        i = 1
        n = 4

        ms1, cov1 = self._f_Si_moments[1]
        ms2, cov2 = self._f_Si_moments[2]
        ms3, cov3 = self._f_Si_moments[3]

        ms1t, cov1t = _get_mat_T_moments(ms1, cov1)
        ms2t, cov2t = _get_mat_T_moments(ms2, cov2)
        ms3t, cov3t = _get_mat_T_moments(ms3, cov3)
        
        expr0 = _Exp_mat_quad(ms3t.reshape(2, 2), cov3t, Dinv)
        expr1 = _Exp_mat_quad(ms2t.reshape(2, 2), cov2t, expr0)
        expr2 = _Exp_mat_quad(ms1t.reshape(2, 2), cov1t, expr1)

        ex = self._f_Ex[i]
        exxt = self._f_Exxt[i]

#        print(exxt*expr1[0, 1])
        EP14 = np.dot(ms3.reshape(2, 2), ms2.reshape(2, 2))
        print("===============")
        yn = self.forward_data[self._f_y_ind_map[n], :]
        EX = block_diag(*(ex[None, :] for k in range(2)))
        b = np.dot(yn, np.dot(Dinv, np.dot(EP14, EX)))
        print(b)
        print("-------")

        _parse_Yn_contrib_for_psi_i(i, n,
                                 yn,
                                 self.forward_dts[i],
                                 self._f_Ex[i],
                                 self._f_Exxt[i],
                                 self._f_Si_moments,
                                 Dinv,
                                 self.A, 2, True)
        

    def _update_psi_i2(self, i, direc):

        # Prior component update - slow version
        _get_prior_cond_pars(self)
        
        if direc=="forward":
            nb = self._N_b_psi
            data_inds = self.forward_data_inds
            data_ind_map = self._f_y_ind_map
            Y = self.forward_data

            dti = self.forward_dts[i]
            Exi = self._f_Ex[i]
            ExixiT = self._f_Exxt[i]

            S_moments = self._f_Si_moments

        else:
            nb = 0
            data_inds = self.backward_data_inds
            data_ind_map = self._b_y_ind_map
            Y = self.backward_data

            dti = self.backward_dts[i]
            Exi = self._b_Ex[i]
            ExixiT = self._b_Exxt[i]

            S_moments = self._b_Si_moments
            

    def _update_psi_i(self, i, direc="forward"):

        # Contribution from the prior
        if direc=="forward":
            nb = self._N_b_psi
            data_inds = self.forward_data_inds
            data_ind_map = self._f_y_ind_map
            Y = self.forward_data

            dti = self.forward_dts[i]
            Exi = self._f_Ex[i]
            ExixiT = self._f_Exxt[i]

            S_moments = self._f_Si_moments

        else:
            nb = 0
            data_inds = self.backward_data_inds
            data_ind_map = self._b_y_ind_map
            Y = self.backward_data

            dti = self.backward_dts[i]
            Exi = self._b_Ex[i]
            ExixiT = self._b_Exxt[i]

            S_moments = self._b_Si_moments

        vec = np.column_stack((
            v[nb+i] for v in self._cond_transforms))

        # E_psi_n_i
        E_psi = np.row_stack((self._backward_psi.mean,
                              self._forward_psi.mean))
        n_i = [j for j in range(E_psi.shape[0]) if j != nb+i]

        # parse the contribution for the prior
        m_prior = np.array([np.dot(v, E_psi_r)
                            for v, E_psi_r in zip(vec.T, E_psi[n_i, :].T)])

        psi_mean = []
        psi_b = np.array(self._backward_psi.mean)
        psi_f = np.array(self._forward_psi.mean)
        psi_full = np.row_stack((psi_b, psi_f))

        bta = self.backward_full_ts[:-1]
        btb = self.backward_full_ts[1:]
        fta = self.forward_full_ts[:-1]
        ftb = self.forward_full_ts[1:]

        bTa, bSa = np.meshgrid(bta, bta)
        bTb, bSb = np.meshgrid(btb, btb)
        fTa, fSa = np.meshgrid(fta, fta)
        fTb, fSb = np.meshgrid(ftb, ftb)
        bfTa, bfSa = np.meshgrid(fta, bta)
        bfTb, bfSb = np.meshgrid(ftb, btb)

        ci_prior = []
        m_prior = []
        for r in range(self._R):
            Cbb = self._cov_funcs["J1J1"](bSb.ravel(), bTb.ravel(),
                                          bSa.ravel(), bTa.ravel(),
                                          1., self.lscales[r]).reshape(bSb.shape)

            Cbf = self._cov_funcs["J1J1"](bfSb.ravel(), bfTb.ravel(),
                                          bfSa.ravel(), bfTa.ravel(),
                                          1., self.lscales[r]).reshape(bfSb.shape)
            
            Cff = self._cov_funcs["J1J1"](fSb.ravel(), fTb.ravel(),
                                          fSa.ravel(), fTa.ravel(),
                                          1., self.lscales[r]).reshape(fSb.shape)            

            C = np.row_stack((np.column_stack((Cbb, Cbf)),
                              np.column_stack((Cbf.T, Cff))))

            ind = [nb+i]
            nind = [j for j in range(C.shape[0]) if j != nb+i]

            C11 = C[nb+i, nb+i]
            C12 = C[nb+i, nind]
            C22 = C[nind, :]
            C22 = C22[:, nind]

            a = psi_full[nind, r]

            try:
                L = np.linalg.cholesky(C22)
            except:
                C22 += np.diag(1e-4*np.ones(C22.shape[0]))
                L = np.linalg.cholesky(C22)
                
            La = np.linalg.solve(L.T, np.linalg.solve(L, a))
            var = C11 - np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, C12.T)))
            m = np.dot(C12, La)
            ci_prior.append(1/var)
            m_prior.append(m)
        E_inv_gp_scale = self._gp_scale_alphas/self._gp_scale_betas
#        print('=======')
#        print(np.diag(E_inv_gp_scale*ci_prior))
#        print('=======')        
        ci_prior = np.diag(E_inv_gp_scale*ci_prior)


#        print("Hmm:",E_inv_gp_scale)
#        ci_prior = np.diag(E_inv_gp_scale/self._cond_vars[:, nb+i])

        ms = [m_prior]
        cis = [ci_prior]

        print("prior contrib")
        print(np.array(cis))

        ###
        Dinv = np.diag(self.obs_noise_pars[0]/self.obs_noise_pars[1])
        for n in reversed(data_inds[data_inds > i]):

            yn = Y[data_ind_map[n]]
            m, ci = _parse_Yn_contrib_for_psi_i(i, n, yn,
                                                dti, Exi, ExixiT,
                                                S_moments, Dinv,
                                                self.A, self._K)

            ms.append(m)
            cis.append(ci)

        m, c = _prod_norm_pars(ms, cis)
        _update_psi_moments(self, i, direc, m, c)

    def _update_gp_scale_pars(self):
        a0 = np.array([50., 50.])
        b0 = np.array([0.5, 0.5])

        # Forward
        psi_m = np.array([m
                          for m in chain(self._backward_psi.mean,
                                           self._forward_psi.mean)])
        psi_var = np.array([np.diag(c)
                            for c in chain(self._backward_psi.cov,
                                           self._forward_psi.cov)])
        
        Nr = psi_m.shape[0]

        alphas = Nr/2 + a0

        ta = np.concatenate((self.backward_full_ts[:-1],
                             self.forward_full_ts[:-1]))
        tb = np.concatenate((self.backward_full_ts[1:],
                             self.forward_full_ts[1:]))

        def _get_cov_pq(p, q, ta, tb, r):
            key = "J{}J{}".format(p, q)
            Ta, Sa = np.meshgrid(ta, ta)
            Tb, Sb = np.meshgrid(tb, tb)
            return self._cov_funcs[key](Sb.ravel(), Tb.ravel(),
                                        Sa.ravel(), Ta.ravel(),
                                        1., self.lscales[r]).reshape(Ta.shape)
        betas = []
        for r in range(self._R):

            C = _get_cov_pq(1, 1, ta, tb, r)
            C += np.diag(1e-6*np.ones(C.shape[0]))
            C0inv = np.linalg.inv(C)
#            print(np.diag(C0inv))
            val1 = np.dot(psi_var[:, r], np.diag(C0inv))
            val2 = np.dot(psi_m[:, r], np.dot(C0inv, psi_m[:, r]))
            print("Values", val1,  val2)
            betas.append(val1 + val2)
        betas = 0.5*np.array(betas) + b0
        
        assert(all(betas > 0))

        print("alphas:", alphas)
        print("betas:", betas)
        self._gp_scale_alphas = alphas
        self._gp_scale_betas = betas

    def _update_psi_moments(self, i, direc, mean, cov):
        _update_psi_moments(self, i, direc, mean, cov)

    """
    Initalisation of the variational parameters
    """
    def _init_psi_distributions(self):
        from scipy.integrate import quad
        f = []
        for _ta, _tb in zip(self.backward_full_ts[:-1],
                            self.backward_full_ts[1:]):
            J1 = quad(lambda t: np.cos(t), _ta, _tb)[0]
            J2 = quad(lambda t: np.exp(-0.5*(t-2)**2), _ta, _tb)[0]
            f.append([J1, J2])
        f = [np.zeros(self._R) for n in range(self._N_b_psi)]
        scale = 0.01
        self._backward_psi = PsiMoments(np.array(f),
                                        [scale*np.diag(np.ones(self._R))
                                         for n in range(self._N_b_psi)])

        f = []
        for _ta, _tb in zip(self.forward_full_ts[:-1],
                            self.forward_full_ts[1:]):
            J1 = quad(lambda t: np.cos(t), _ta, _tb)[0]
            J2 = quad(lambda t: np.exp(-0.5*(t-2)**2), _ta, _tb)[0]
            f.append([J1, J2])
        f = [np.zeros(self._R) for n in range(self._N_f_psi)]

        self._forward_psi = PsiMoments(np.array(f),
                                       [scale*np.diag(np.ones(self._R))
                                        for n in range(self._N_f_psi)])

        self._f_Si_moments = [[] for i in range(self._N_f_psi)]
        for i in range(self._N_f_psi):
            _store_Si_moments(self, i, "forward")

        self._b_Si_moments = [[] for i in range(self._N_b_psi)]
        for i in range(self._N_b_psi):
            _store_Si_moments(self, i, "backward")


    def pred_latent_force(self, r, pred_times,
                          b_psi, f_psi, return_var=False):

        Cbb = self._Cbbs[r]
        Cbf = self._Cbfs[r]
        Cff = self._Cffs[r]

        b_ta = self.backward_full_ts[:-1]
        b_tb = self.backward_full_ts[1:]
        f_ta = self.forward_full_ts[:-1]
        f_tb = self.forward_full_ts[1:]

        ta = np.concatenate((b_ta, f_ta))
        tb = np.concatenate((b_tb, f_tb))

        Ta, Sa = np.meshgrid(ta, pred_times)
        Tb, Sb = np.meshgrid(tb, pred_times)

        C22 = np.row_stack((np.column_stack((Cbb, Cbf)),     # Probably a neater
                            np.column_stack((Cbf.T, Cff))))  # way to invert just useing the blocks

        l = self.lscales[r]
        C12 = self._cov_funcs["J0J1"](Sb.ravel(), Tb.ravel(),
                                      None, Ta.ravel(),
                                      1, l).reshape(Ta.shape)
        try:
            L = np.linalg.cholesky(C22)
        except:
            C22 += np.diag(1e-4*np.ones(C22.shape[0]))
            L = np.linalg.cholesky(C22)
        
        psi = np.concatenate((b_psi, f_psi))

        mean = np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, psi)))

        if return_var:
            S1, S2 = np.meshgrid(pred_times, pred_times)
            C11 = self._cov_funcs["J0J0"](S1.ravel(), S2.ravel(),
                                          None, None,
                                          1.,
                                          1.).reshape(S1.shape)
            var = C11 - np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, C12.T)))
            return mean, var
        
        return mean 


##
# Covariance matrix is structured as
# 
# 


"""

Updating functions for the different variational parameters

"""
def _update_psi_i(obj, i, direc="forward"):

    if direc == "forward":
        ns = [n for n in obj.forward_data_inds if n > i]
    elif direc == "backward":
        ns = [n for n in obj.backward_data_inds if n > i]

    # Take the expectation of the diagonal inv. cov. matrix under
    # the variational parameters of the noise

    # Since each sigma^2 follows an inverse gamma then we have
    # E[1/sigma^2] = a/b 
    E_Obs_Inv_Cov = np.diag(obj.obs_noise_pars[:, 0]/ obj.obs_noise_pars[:, 1])


###
# For the variable
#
#     P_{i,n} = S(ψ_{n-1}) ... S(ψ_{i+1})
#
# returns E[P_{i, n}] based on the current moments of
# the variational parameters ψ
###
"""
def _Exp_Pin(obj, i, n, direc):
    
    if direc == "forward":
        Si_moments = obj._f_Si_moments
        dts = obj.forward_dts
    elif direc == "backward":
        Si_moments = obj._b_Si_moments
        dts = obj.backward_dts

    res = np.diag(np.ones(self._K))
    for ind in range(i+1, n):
        Sind_mean = Si_moments[ind][0]
        res = np.dot(Sind_mean.reshape(self._K, self._K), res)

    return res
"""

"""
    
Storing and updating of internal parameters related to the
variational fitting

"""

def _init_ex_exxt(obj, Ex0, Ex0x0T):

    # Initalise some containers for moments
    obj._b_Ex = [[] for n in range(obj._N_b_psi+1)]
    obj._b_Exxt = [[] for n in range(obj._N_b_psi+1)]

    obj._f_Ex = [[] for n in range(obj._N_f_psi+1)]
    obj._f_Exxt = [[] for n in range(obj._N_f_psi+1)]

    for direc in ["backward", "forward"]:
        _update_ex_exxt(obj, 0, direc, Ex0, Ex0x0T)


############################
#
# E[X_{i+1}] = E[S_i]E[x_i]
#
def _update_ex_exxt(obj, i, direc="forward", Ex0=None, Ex0x0T=None):
    assert(not(i == 0 and Ex0 is None))

    if direc == "forward":
        N = obj._N_f_psi
        _Si_moments = obj._f_Si_moments
        Ex_list = obj._f_Ex
        Exxt_list = obj._f_Exxt
    elif direc == "backward":
        N = obj._N_b_psi
        _Si_moments = obj._b_Si_moments
        Ex_list = obj._b_Ex
        Exxt_list = obj._b_Exxt

    if i == 0:
        Ex_list[i] = Ex0
        Exxt_list[i] = Ex0x0T
        for n in range(i+1, N+1):
            evs, evs_cov = _Si_moments[n-1]
            eS = evs.reshape(obj._K, obj._K)
        
            Ex_list[n] = np.dot(eS, Ex_list[n-1])
            Exxt_list[n] = _Exp_mat_quad(eS, evs_cov, Exxt_list[n-1])   

    else:
        for n in range(i, N+1):
            # mean and covar of vec(Si)
        
            evs, evs_cov = _Si_moments[n-1]
            eS = evs.reshape(obj._K, obj._K)
        
            Ex_list[n] = np.dot(eS, Ex_list[n-1])
            Exxt_list[n] = _Exp_mat_quad(eS, evs_cov, Exxt_list[n-1])

def _update_ex_exxt2(obj, i, direc="forward", Ex0=None, Ex0x0T=None):

    if direc == "forward":
        N = obj._N_f_psi + 1
        _Si_moments = obj._f_Si_moments
        Ex_list = obj._f_Ex
        Exxt_list = obj._f_Exxt

    elif direc == "backward":
        N = obj._N_b_psi + 1
        _Si_moments = obj._b_Si_moments
        Ex_list = obj._b_Ex
        Exxt_list = obj._b_Exxt
    
    if i == 0:
        Ex_list[i] = Ex0
        Exxt_list[i] = Ex0x0T
        i = i+1

    for n in range(i, N):
        evs, evs_cov = _Si_moments[n-1]
        eS = evs.reshape(obj._K, obj._K)

        Ex_list[n] = np.dot(eS, Ex_list[n-1])
        Exxt_list[n] = _Exp_mat_quad(eS, evs_cov, Exxt_list[n-1])

        

def _update_psi_moments(obj, i, direc, mean, cov, ex0=None, ex0x0T=None):
    if direc == "forward":
        psi_moms = obj._forward_psi
    elif direc == "backward":
        psi_moms = obj._backward_psi

    psi_moms.mean[i] = mean
    psi_moms.cov[i] = cov

    _store_Si_moments(obj, i, direc)
#    _init_ex_exxt(obj, obj._f_Ex[0], obj._f_Exxt[0])
    _update_ex_exxt2(obj, i+1, direc)



def _store_Si_moments(obj, i, direc="forward"):
    if direc == "forward":
        psi_i_mean = obj._forward_psi.mean[i]
        psi_i_cov = obj._forward_psi.cov[i]
        
        dt = obj.forward_dts[i]
        obj._f_Si_moments[i] = _get_vec_Si_pars(psi_i_mean, psi_i_cov,
                                                obj.A, dt)
    elif direc == "backward":
        psi_i_mean = obj._backward_psi.mean[i]
        psi_i_cov = obj._backward_psi.cov[i]

        dt = obj.backward_dts[i]
        obj._b_Si_moments[i] = _get_vec_Si_pars(psi_i_mean, psi_i_cov,
                                                obj.A, dt) 



    

"""
Model Utility functions
"""

##
# returns the mean and covariance of the variables
#
#     S = I + A*dt + sum_r A_r * psi_i_r
#
# given the mean and covariance of the variational
# parameters psi_i
def _get_vec_Si_pars(psi_i_mean, psi_i_cov, A, dt):

    R = psi_i_mean.size
    K = A[0].shape[0]
    I = np.diag(np.ones(K))

    EvecSi = sum([A[i+1].ravel()*mi
                  for i, mi in enumerate(psi_i_mean)])
    EvecSi += I.ravel() + dt*A[0].ravel()
    
    Cov_vecSi = np.zeros((K*K, K*K))
    for s in range(R):
        _as = A[s+1].ravel()
        for t in range(R):
            _at = A[t+1].ravel()
            Cov_vecSi += psi_i_cov[s, t]*np.outer(_as, _at)

    return EvecSi, Cov_vecSi

###
# The variable P(i, n) is defined by
#
#     P(i, n) = S(n-1)S(n-2)···S(i+1) 
#
# from the decomposition x(n) = P(i, n)S(i)x[i]

# 𝔼[Pin(n, i)]
def _Exp_Pin(i, n, S_moments, K):
    res = np.diag(np.ones(K))
    for j in range(i+1, n):
        res = np.dot(S_moments[j][0].reshape(K, K), res)
    return res

# 𝔼[Pin(n, i)T Q Pin(n, i)]
def _Exp_Pin_mat_quad(i, n, S_moments, Q, K):
    res = Q
    for m, c in reversed(S_moments[i+1:n]):
        # need to get the moments of S transpose
        m, c = _get_mat_T_moments(m, c)
        res = _Exp_mat_quad(m.reshape(K, K), c, res)
    return res


##
# (yn-xn)^T Dinv (yn-xn)
#
# Dinv = 𝔼[ ]
def _parse_Yn_contrib_for_psi_i(i, n,
                                yn, dti, Exi, ExixiT,
                                S_moments, Dinv,
                                A, K, print_it=False):

    Exp_Pin = _Exp_Pin(i, n, S_moments, K)
    Exp_PinT_D_Pin = _Exp_Pin_mat_quad(i, n, S_moments, Dinv, K)

    Si_inv_cov = np.row_stack((
        np.column_stack((ExixiT*Mij for Mij in row))
        for row in Exp_PinT_D_Pin[:, ]))

    _EX = scipy.linalg.block_diag(*(Exi[None, :] for k in range(K)))
    Exp_B = np.dot(Dinv, np.dot(Exp_Pin, _EX))
    y_Exp_B = np.dot(yn, Exp_B)

    # Note the pseudo-inverse
    try:
        Si_mean = np.linalg.solve(Si_inv_cov, y_Exp_B)
    except:
        Si_mean = np.dot(np.linalg.pinv(Si_inv_cov), y_Exp_B)

    return _parse_Si_m_inv_cov(A, Si_mean, Si_inv_cov, dti, K)

def _parse_Si_m_inv_cov(A, Si_mean, Si_inv_cov, dti, K):
    Avecs = np.column_stack((a.ravel() for a in A[1:]))
    b = (np.diag(np.ones(K)) + A[0]*dti).ravel()

    psi_m, psi_inv_cov = mvt_linear_trans(Avecs, b,
                                          Si_mean, inv_cov=Si_inv_cov)
    return psi_m, psi_inv_cov

"""
Matrix random variable utility
"""
##
# if X is a matrix random variable such that
#
# 𝔼[vec(X)] = vec_mean,    Cov[vec(X)] = vec_cov
#
# then returns the mean and variance of vec(X.T) by
# swapping indices of the mean and columns of the cov matrix
def _get_mat_T_moments(vec_mean, vec_cov):
    N = vec_cov.shape[0]
    rootN = int(np.sqrt(N))
    ind_mat = np.array(range(N)).reshape(rootN, rootN)
    t_ind = ind_mat.T.ravel()

    mean = vec_mean[t_ind]

    cov = vec_cov[:, t_ind]
    cov = cov[t_ind, :]

    return mean, cov

###
# 𝔼[ X M X.T ]
#
# cov_X is the N^2 x N^2 matrix with
#
# where vec(X) has covariance matrix cov_X
def _Exp_mat_quad(mean_X, cov_X, M):
    _N = M.shape[0]
    res = np.zeros((_N, _N))
    for m in range(_N):
        for n in range(_N):
            cov_xm_xn = cov_X[m*_N:(m+1)*_N, n*_N:(n+1)*_N]
            expr1 = np.dot(cov_xm_xn, M)
            res[m, n] = np.trace(expr1) 
    res += np.dot(mean_X, np.dot(M, mean_X.T))
    return res

#####
#
# if X ~ N(mean, cov) and X = Az + b then we
# rearrange for the mean and covariance of z
#
# (possibly degenerate)
def mvt_linear_trans(A, b, mean, cov=None, inv_cov=None, return_type="inv"):
    if inv_cov is None:
        inv_cov = np.linalg.inv(cov)
        
    z_inv_cov = np.dot(A.T, np.dot(inv_cov, A))

    w = np.dot(A.T, np.dot(inv_cov, mean - b))
    z_mean = np.linalg.solve(z_inv_cov, w)

    if return_type == "inv":
        return z_mean, z_inv_cov
    else:
        return z_mean, np.linalg.inv(z_inv_cov)

"""
Updating the gp scale parameters
"""

########################################################
#
#     ln p(a | g ) propto -(1/a)*0.5*psi^T C0inv psi - (N/2)ln(a) + ln p(a)
#
def _update_gp_scale_pars(obj):

    # Backward contribution
    b_psi_m = np.array(obj._backward_psi.mean)
    b_psi_v = np.array([np.diag(c) for c in obj._backward_psi.cov])

    # Forward contribution
    f_psi_m = np.array(obj._forward_psi.mean)
    f_psi_v = np.array([np.diag(c) for c in obj._forward_psi.cov])

    psi_m = np.row_stack((b_psi_m, f_psi_m))
    psi_var = np.row_stack((b_psi_v, f_psi_v))

    alphas = psi_m.shape[0]/2

    betas = []
    for r in range(obj._R):

        Cr_inv = obj._Cr_invs[r]
        val1 = np.trace(np.dot(np.diag(psi_var[:, r]), Cr_inv))
        val2 = np.dot(psi_m[:, r], np.dot(Cr_inv, psi_m[:, r]))

        betas.append(val1+val2)
    betas = np.array(betas)

    
def _setup_times(obj, t0_ind, data_times, data_Y, step_size):

    # Store data
    obj.backward_data = np.flip(data_Y[:t0_ind+1], 0)
    obj.forward_data = data_Y[t0_ind:]

    # handle forward times, all those times occuring on and after t0
    post_times = data_times[t0_ind:]

    forward_full_ts = [post_times[0]]
    forward_data_inds = [0]

    for ta, tb in zip(post_times[:-1], post_times[1:]):
        _n = np.ceil((tb-ta)/step_size)
        _ts = np.linspace(ta, tb, _n)
        forward_full_ts = np.concatenate((forward_full_ts, _ts[1:]))
        forward_data_inds.append(len(forward_full_ts)-1)

    obj.forward_full_ts = forward_full_ts
    obj.forward_dts = np.diff(forward_full_ts)
    obj.forward_data_inds = np.array(forward_data_inds)
    obj._N_f_psi = obj.forward_dts.size
                     
    # handle backward times, all those times up to *and including* t0
    pre_times = data_times[:t0_ind+1]    
    backward_times = [t for t in reversed(pre_times)]
    backward_full_ts = [backward_times[0]]
    backward_data_inds = [0]
    for ta, tb in zip(backward_times[:-1], backward_times[1:]):
        _n = np.ceil((ta-tb)/step_size)
        _ts = np.linspace(ta, tb, _n)
        backward_full_ts = np.concatenate((backward_full_ts, _ts[1:]))
        backward_data_inds.append(len(backward_full_ts)-1)
    obj.backward_full_ts = backward_full_ts
    obj.backward_dts = np.diff(backward_full_ts)
    obj.backward_data_inds = np.array(backward_data_inds)
    obj._N_b_psi = obj.backward_dts.size


    # After augmenting the time vectors to create the grid of knot points for
    # the model we store each of the time point indices to which there is a
    # corresponding data observations in
    #
    # .backward_data_inds and .forward_data_inds
    #
    # the values of these observations may then be obtained from
    #
    # Y[n(i)] = backward_data [._b_y_ind_map[i] ]
    obj._b_y_ind_map = {n: i for i, n in enumerate(obj.backward_data_inds)}
    obj._f_y_ind_map = {n: i for i, n in enumerate(obj.forward_data_inds)}        


####
#
# Since the prior factors as
#
#  p(ψ_i | ψ_{-i}) 
# 
def _handle_cov_setup(obj):

    b_ta = obj.backward_full_ts[:-1]
    b_tb = obj.backward_full_ts[1:]
    f_ta = obj.forward_full_ts[:-1]
    f_tb = obj.forward_full_ts[1:]

    # Prepare times for ravelling
    b_Ta, b_Sa = np.meshgrid(b_ta, b_ta)
    b_Tb, b_Sb = np.meshgrid(b_tb, b_tb)
    f_Ta, f_Sa = np.meshgrid(f_ta, f_ta)
    f_Tb, f_Sb = np.meshgrid(f_tb, f_tb)

    bf_Ta, bf_Sa = np.meshgrid(f_ta, b_ta)
    bf_Tb, bf_Sb = np.meshgrid(f_tb, b_tb)


    _Cbbs = []
    _Cffs = []
    _Cbfs = []
    _Cr_invs = []

    # If the spacing between observations could be homogenous
    # then these calculations could be simplified
    _cond_transforms = []
    _cond_vars = []

    tavec = np.concatenate((b_ta, f_ta))
    tbvec = np.concatenate((b_tb, f_tb))
    print(tbvec-tavec)
    
    for l in obj.lscales:
        Cbb = _cov_funcs["J1J1"](b_Sb.ravel(), b_Tb.ravel(),
                                 b_Sa.ravel(), b_Ta.ravel(),
                                 1., l).reshape(b_Ta.shape)

        Cff = _cov_funcs["J1J1"](f_Sb.ravel(), f_Tb.ravel(),
                                 f_Sa.ravel(), f_Ta.ravel(),
                                 1., l).reshape(f_Ta.shape)

        Cbf = _cov_funcs["J1J1"](bf_Sb.ravel(), bf_Tb.ravel(),
                                 bf_Sa.ravel(), bf_Ta.ravel(),
                                 1., l).reshape(bf_Ta.shape)

        _Cbbs.append(Cbb)
        _Cbfs.append(Cbf)
        _Cffs.append(Cff)

        C = np.row_stack((np.column_stack((Cbb, Cbf)),
                          np.column_stack((Cbf.T, Cff))))

        C2 = np.zeros((tavec.size, tavec.size))
        for i in range(tavec.size):
            for j in range(tavec.size):
                C2[i, j] = _cov_funcs["J1J1"](tbvec[i], tbvec[j],
                                              tavec[i], tavec[j],
                                              1., l)
        _Cr_invs.append(np.linalg.inv(C))

        # Store copies of the transformation vectors for
        # getting the conditional mean of each psi
        T = []
        cond_V = []
        for i in range(C.shape[0]):
            row_idx = np.array([j for j in range(C.shape[0])
                                if j != i])
            col_idx = row_idx.copy()

            c11 = C[i, i]
            c12 = C[i, col_idx]            
            c22 = C[row_idx[:, None], col_idx]
            c22inv = np.linalg.inv(c22)

            cond_V.append(c11 - np.dot(c12, np.dot(c22inv, c12.T)))
            T.append(np.dot(c12, c22inv))

        _cond_transforms.append(np.array(T))
        _cond_vars.append(cond_V)

    # Attach the cov functions for later use
    obj._Cbbs = _Cbbs
    obj._Cbfs = _Cbfs
    obj._Cffs = _Cffs
    obj._cov_funcs = _cov_funcs
    obj._Cr_invs = _Cr_invs

    obj._cond_transforms = _cond_transforms
    obj._cond_vars = np.array(_cond_vars)
    
# Load the squared exponential covariance functions


##
# Constants
rootPi = np.sqrt(np.pi)

###########
#                               
# ∫erf(z)dz = z erf(z) + exp(-z**2)/sqrt(π)
def integral_erf(lower, upper):
    zu = upper*erf(upper) + np.exp(-upper**2)/rootPi
    zl = lower*erf(lower) + np.exp(-lower**2)/rootPi
    return zu-zl

##
# /  /
# |  |
# |  | theta0*exp(-theta1**2*(s-t) ds dt
# /  /
#
def func(sb, tb, sa, ta, theta0, theta1):
    I1 = -integral_erf(theta1*(sb-ta), theta1*(sb-tb))
    I2 = -integral_erf(theta1*(sa-ta), theta1*(sa-tb))
    C = theta0*0.5*rootPi/(theta1**2)

    return theta0*C*(I1 - I2)

###
# /tb
# |
# | theta0*exp(-theta1**2*(sb-t)**2)dt
# /ta
#
def single_int(sb, tb, sa, ta, theta0, theta1):
    I1 = erf(theta1*(tb-sb))
    I2 = erf(theta1*(ta-sb))
    return 0.5*theta0*rootPi*(I1 - I2)/theta1

def kse(sb, tb, sa, ta, theta0, theta1):
    return np.exp(-theta1**2*(sb-tb)**2)*theta0**2

_cov_funcs = {
    "J0J0": kse,
    "J0J1": single_int,
    "J1J1": func
    }



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

"""
New Stuff!
"""
def _get_cov_pq(p, q, ta, tb, cfunc, theta0, theta1):
    key = "J{}J{}".format(p, q)

    Ta, Sa = np.meshgrid(ta, ta)
    Tb, Sb = np.meshgrid(tb, tb)

    return cfunc[key](Sb.ravel(), Tb.ravel(),
                      Sa.ravel(), Ta.ravel(),
                      theta0, theta1).reshape(Ta.shape)

def _get_cov_partition(i, N, A, B, C):

    n_ind = np.array([j for j in range(N) if j != i])

    M11 = np.array([[A[i, i], B[i, i]],
                    [B[i, i], C[i, i]]])

    M12 = np.row_stack((
        np.concatenate((A[i, n_ind], B[i, n_ind])),
        np.concatenate((B.T[i, n_ind], C[i, n_ind]))))

    M22 = np.row_stack((
        np.column_stack((A[n_ind[:, None], n_ind], B[n_ind[:, None], n_ind])),
        np.column_stack((B[n_ind[:, None], n_ind].T, C[n_ind[:, None], n_ind]))))

    return M11, M12, np.linalg.cholesky(M22)
        
def _back_sub(L, x):
    return np.linalg.solve(L.T, np.linalg.solve(L, x))


def _get_prior_cond_pars(i, obj):
    from scipy.integrate import quad
    def _g1(s):
        return np.cos(s)
    def _g2(s):
        return np.sin(s)

    ta = np.concatenate((obj.backward_full_ts[:-1],
                         obj.forward_full_ts[:-1]))
    tb = np.concatenate((obj.backward_full_ts[1:],
                         obj.forward_full_ts[1:]))

    Epsi = np.column_stack((
        _g1(tb), 
        _g2(tb),
        np.sin(tb)-np.sin(ta),
        -np.cos(tb)+np.cos(ta)))

#    Epsi = np.row_stack((
#        np.array(obj._backward_psi.mean),
#        np.array(obj._forward_psi.mean)))

    nEpsi = Epsi[[j for j in range(ta.size) if j != i], :]    
    nEpsi0 = nEpsi[:, :obj._R]
    nEpsi1 = nEpsi[:, obj._R:]

    print(nEpsi)

    m0_list = []
    m1_list = []
    for r in range(obj._R):
        l = obj.lscales[r]

        C00 = _get_cov_pq(0, 0, ta, tb, obj._cov_funcs, 1., l)
        C01 = _get_cov_pq(0, 1, ta, tb, obj._cov_funcs, 1., l)
        C11 = _get_cov_pq(1, 1, ta, tb, obj._cov_funcs, 1., l)

        Caa, Cab, Lbb = _get_cov_partition(i, ta.size, C00, C01, C11)
        Ccond = Caa - np.dot(Cab, _back_sub(Lbb, Cab.T))

        v0 = nEpsi0[:, r]
        v1 = nEpsi1[:, r]
        v = np.column_stack((v0, v1))
        v = v.ravel()
        v = np.concatenate((v0, v1))

        mcond = np.dot(Cab, _back_sub(Lbb, v))
        m0_list.append(mcond[0])
        m1_list.append(mcond[1])


    return tb[i], np.array(m0_list)
