import numpy as np
from collections import namedtuple
from .nestedgpintegrals import ngpintegrals_sqexpcovs
from scipy.stats import norm
from scipy.linalg import block_diag
from scipy.special import erf


PsiMoments = namedtuple("PsiMoments", "mean cov")


class VariationalMLFM2:

    def __init__(self,
                 t0_ind, data_times, data_Y,
                 step_size,
                 As=None,
                 lscales=None,
                 obs_noise_priors=None):

        self.A = np.asarray(As)
        self._R = len(As)-1
        self.lscales = lscales

        # Initalise the priors for the scale
        self.obs_noise_pars = np.asarray(obs_noise_priors)

        _setup_times(self, t0_ind, data_times, data_Y, step_size)
        _handle_cov_setup(self)        

        # Initalise the trajectory variational parameters
        self._init_psi_distributions()

        _update_gp_scale_pars(self)


    """
    Initalisation of the variational parameters
    """
    def _init_psi_distributions(self):

        self._backward_psi = PsiMoments([np.zeros(self._R)
                                         for n in range(self._N_b_psi)],
                                        [np.diag(np.ones(self._R))
                                         for n in range(self._N_b_psi)])


        self._forward_psi = PsiMoments([np.zeros(self._R)
                                        for n in range(self._N_f_psi)],
                                       [np.diag(np.ones(self._R))
                                        for n in range(self._N_f_psi)])

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
        print(np.linalg.eig(C22)[0])
        C12 = self._cov_funcs["J0J1"](Sb.ravel(), Tb.ravel(),
                                      None, Ta.ravel(),
                                      1., self.lscales[r]).reshape(Ta.shape)
        L = np.linalg.cholesky(C22)
        psi = np.concatenate((b_psi, f_psi))
        
        mean = np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, psi)))

        if return_var:
            S1, S2 = np.meshgrid(pred_times, pred_times)
            C11 = self._cov_funcs["J0J0"](S1.ravel(), S2.ravel(),
                                          None, None,
                                          1., self.lscales[r]).reshape(S1.shape)
            var = C11 - np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, C12.T)))
            return mean, var
        
        return mean 



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
    
Storing and updating of internal parameters related to the
variational fitting

"""
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

    
    print(alphas/betas)



def _setup_times(obj, t0_ind, data_times, data_Y, step_size):

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
    obj.forward_data_inds = forward_data_inds
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
    obj.backward_data_inds = backward_data_inds
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


_cov_funcs = {"J0J1": single_int,
             "J1J1": func}

