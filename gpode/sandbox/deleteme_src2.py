import numpy as np
import scipy.linalg
from scipy.integrate import dblquad


def Ik(ta, tb, inv_lscale):
    C = np.zeros((ta.size, ta.size))
    for i in range(ta.size):
        for j in range(i+1):
            C[i, j] = dblquad(lambda y, x: np.exp(-inv_lscale*(x-y)**2),
                              ta[i], tb[i],
                              lambda x: ta[j], lambda x: tb[j])[0]
            C[j, i] = C[i, j]
    return C


class MyObj: 
    def __init__(self,
                 x0,
                 pre_times, pre_data,                 
                 post_times, post_data,
                 step_size, A):

        try:
            assert(pre_times[-1] == post_times[0])
            self._t0 = pre_times[-1]
        except Exception as e:
            raise type(e)

        self.x0 = x0
        self.A = A

        self._R = A.shape[0]-1
        self._K = A.shape[1]

        self.forward_data = post_data
        self.backward_data = np.flip(pre_data, 0)
        
        # Processing of discretisation of the solution,
        # the step size is handled approximately

        # reverse the times before t0
        backward_times = [t for t in reversed(pre_times)]
        backward_full_ts = [backward_times[0]]
        backward_data_inds = [0]
        for ta, tb in zip(backward_times[:-1], backward_times[1:]):
            _n = np.ceil((ta-tb)/step_size)
            _ts = np.linspace(ta, tb, _n)
            backward_full_ts = np.concatenate((backward_full_ts, _ts[1:]))
            backward_data_inds.append(len(backward_full_ts)-1)
        self.backward_full_ts = backward_full_ts
        self.backward_dts = np.diff(backward_full_ts)
        self.backward_data_inds = backward_data_inds
        self._N_b_psi = self.backward_dts.size

        # Forward times
        forward_full_ts = [post_times[0]]
        forward_data_inds = [0]
        for ta, tb in zip(post_times[:-1], post_times[1:]):
            _n = np.ceil((tb-ta)/step_size)
            _ts = np.linspace(ta, tb, _n)
            forward_full_ts = np.concatenate((forward_full_ts, _ts[1:]))
            forward_data_inds.append(len(forward_full_ts)-1)
        self.forward_full_ts = forward_full_ts
        self.forward_dts = np.diff(forward_full_ts)
        self.forward_data_inds = forward_data_inds
        self._N_f_psi = self.forward_dts.size

        # Initalise the moment of the variational latent GP
        # parameters to mean zero and unit covariance
        self._init_psi_distributions()
        self._init_ex_exxt(x0, np.outer(x0, x0))

        #
        self._b_y_ind_map = {n: i for i, n in enumerate(self.backward_data_inds)}
        self._f_y_ind_map = {n: i for i, n in enumerate(self.forward_data_inds)}        

        #
        self._f_g_covs = [Ik(self.forward_full_ts[:-1],
                             self.forward_full_ts[1:], l)
                          for l in [17., 27.]]
        self._f_g_inv_covs = [np.linalg.inv(c) for c in self._f_g_covs]
    """
    Variational parameters and initialisation of
    """
    def _init_psi_distributions(self):
        
        self.backward_psi_moments = [[np.zeros(self._R), 0.1*np.diag(np.ones(self._R))]
                                     for n in range(self._N_b_psi)]
        
        self.forward_psi_moments = [[np.zeros(self._R), 0.1*np.diag(np.ones(self._R))]
                                    for n in range(self._N_f_psi)]

    # Stores the inital values of Ex and ExxT for 
    def _init_ex_exxt(self, Ex0, Ex0x0T):

        for direc, psi_moms, dts in zip(["b", "f"],
                                        [self.backward_psi_moments, self.forward_psi_moments],
                                        [self.forward_dts, self.backward_dts]):

            Ex_list = [Ex0]
            Exxt_list = [Ex0x0T]

            vec_Si_m_cov = []

            for i, (psi_m, psi_cov) in enumerate(psi_moms):
                # mean and covariance function of vec(Si)
                evs, evs_cov = _get_vec_Si_pars(psi_m, psi_cov,
                                                self.A, self.forward_dts[i])

                # Also store the implied distribution on the fundamental solution
                # approximations Si
                vec_Si_m_cov.append([evs, evs_cov])

                eS = evs.reshape(self._K, self._K)
                ex = np.dot(eS, Ex_list[-1])
                Ex_list.append(ex)

                exxt = _Exp_mat_quad(eS, evs_cov, Exxt_list[-1])
                Exxt_list.append(exxt)

            if direc == "b":
                self._b_Ex = np.array(Ex_list)
                self._b_Exxt = np.array(Exxt_list)
                self._b_Si_moments = vec_Si_m_cov

            else:
                self._f_Ex = np.array(Ex_list)
                self._f_Exxt = np.array(Exxt_list)
                self._f_Si_moments = vec_Si_m_cov


    """
    
    Storing and updating of internal parameters related to the
    variational fitting
    
    """
    def _store_Si_moments(self, i, direc="forward"):
        if direc == "forward":
            psi_i_mean, psi_i_cov = self.forward_psi_moments[i]
            dt = self.forward_dts[i]
            self._f_Si_moments[i] = _get_vec_Si_pars(psi_i_mean, psi_i_cov,
                                                     self.A, dt)
        elif direc == "backward":
            psi_i_mean, psi_i_cov = self.backward_psi_moments[i]
            dt = self.backward_dts[i]
            self._b_Si_moments[i] = _get_vec_Si_pars(psi_i_mean, psi_i_cov,
                                                     self.A, dt)            

    """
    Updating of noise scales
    """
    def _update_noise_pars(self):

        a0 = np.array([11., 11.])
        beta0 = np.array([1., 1.])

        # forward contributions
        Nf = len(self.forward_data_inds)
        alphas = a0 + Nf/2

        betas = np.zeros(self._K)
        for ni in self.forward_data_inds:
            yn = self._f_y_ind_map[ni]
            Exi = self._f_Ex[ni]
            ExxT = self._f_Exxt[ni]

            betas += yn**2 - 2*yn*Exi + np.diag(ExxT)

        betas *= 0.5
        betas += beta0

        self._scale_pars = [alphas, betas]
        self._exp_inv_scale_sq = alphas/betas #betas/(alphas-1)
        print("prior_mean:", beta0/(a0-1))
        print("var mean:", betas/(alphas-1))
        print("1/", self._exp_inv_scale_sq)
            
    def _update_gp_scale_pars(self):

        a0 = np.array([0., 0.])
        b0 = np.array([0., 0.])

        # Forward
        psi_m = np.array([m for m, _ in self.forward_psi_moments])
        psi_var = np.array([np.diag(c) for _, c in self.forward_psi_moments])

        Nr = psi_m.shape[0]

        alphas = Nr/2 + a0

        betas = []
        for r in range(self._R):
            C0inv = self._f_g_inv_covs[r]

            val1 = np.trace(np.dot(np.diag(psi_var[:, r]), C0inv))
            val2 = np.dot(psi_m[:, r], np.dot(C0inv, psi_m[:, r]))

            betas.append(val1+val2)
        betas = 0.5*np.array(betas) + b0

        self._inv_Exp_gp_scale_sq = alphas/betas
    
    """

    Updating of latent integrated GP variational parameters

    """

    def _update_psi_i(self, i, Dinv, direc="forward"):
        if direc == "forward":
            ns = [n for n in self.forward_data_inds if n > i]
        elif direc == "backward":
            ns = [n for n in self.backward_data_inds if n > i]

#        print("i: {} | ns: {}".format(i, ns))

        ms = []
        inv_covs = []
        for n in ns:
            m, ic = self._parse_Yn_contrib_for_psi_i(i, n, Dinv, direc)
            ms.append(m)
            inv_covs.append(ic)
            

        # To do - add the contribution from the prior
        p_m = []
        p_iv = []
        ts = self.forward_full_ts[1:]
        N = ts.size

#        T1, T2 = np.meshgrid(ts, ts)
#        C0 = 2*np.exp(-2.*(T1.ravel()-T2.ravel())**2).reshape(T1.shape)
        a2 = np.array([m for nt,(m, c) in enumerate(self.forward_psi_moments)
                       if nt != i])
        n_ind = [nt for nt in range(N) if nt != i]

        scales = 1./np.array(self._inv_Exp_gp_scale_sq)
        for r, scale in zip(range(self._R), scales):
            C0 = scale*self._f_g_covs[r]
            C22 = C0[n_ind, :]
            C22 = C22[:, n_ind]
            C12 = C0[i, n_ind]

            mc = np.dot(C12, np.dot(np.linalg.inv(C22), a2[:, r]))
            cc = C0[i, i] - np.dot(C12, np.dot(np.linalg.inv(C22), C12.T))

            p_m.append(mc)
            p_iv.append(1/cc)

        ms.append(p_m)
        inv_covs.append(np.diag(p_iv))
            
        m, cov = _prod_norm_pars(ms, inv_covs)
        # Handle updating of all moments of si, xi
        if direc == "forward":
            self.forward_psi_moments[i] = [m, cov]
        elif direc == "backward":
            self.backward_psi_moments[i] = [m, cov]

        self._store_Si_moments(i, direc)

        # Lazy updating of Ex ExxT
        self._init_ex_exxt(self.x0, np.outer(self.x0, self.x0))
              
    def _parse_Yn_contrib_for_psi_i(self, i, n, Dinv, direc):

        if direc == "forward":
            Exi = self._f_Ex[i]
            ExixiT = self._f_Exxt[i]
            yn = self.forward_data[self._f_y_ind_map[n], ]
            dti = self.forward_dts[i]
        elif direc == "backward":
            Exi = self._b_Ex[i]
            ExixiT = self._b_Exxt[i]
            yn = self.backward_data[self._b_y_ind_map[n], ]
            dti = self.backward_dts[i]

        Exp_PinT_D_Pin = self._Exp_Pin_mat_quad(i, n, Dinv, direc)


        Si_inv_cov = np.row_stack((
            np.column_stack((ExixiT*Mij for Mij in row))
            for row in Exp_PinT_D_Pin[:, ]))

        Exp_Pin = self._Exp_Pin(i, n, direc)
        EX = scipy.linalg.block_diag(*(Exi[None, :] for k in range(self._K)))

        Exp_B = np.dot(Dinv, np.dot(Exp_Pin, EX))
        y_Exp_B = np.dot(yn, Exp_B)
        
        Si_mean = np.dot(np.linalg.pinv(Si_inv_cov), y_Exp_B)

#        print("=======")
#        mat = np.dot(EX.T, np.dot(Dinv, EX))
#        print(Si_inv_cov)
#        print(ExixiT)
#        print("========")

        return self._parse_Si_m_inv_cov(i, Si_mean, Si_inv_cov, dti)


    def _parse_Si_m_inv_cov(self, i, Si_mean, Si_inv_cov, dti):
            # Convert these contributions to the values of
        # the random components of psi, (psi_i1,...,psi_iR)
        Avecs = np.column_stack((a.ravel() for a in self.A[1:]))
        b = (np.diag(np.ones(self._K)) + self.A[0]*dti).ravel()

        psi_m, psi_inv_cov = mvt_linear_trans(Avecs, b,
                                              Si_mean, inv_cov=Si_inv_cov)
        return psi_m, psi_inv_cov


    ###
    # For the variable
    #
    #     P_{i,n} = S(ψ_{n-1}) ... S(ψ_{i+1})
    #
    # returns E[P_{i, n}] based on the current moments of
    # the variational parameters ψ
    ###
    def _Exp_Pin(self, i, n, direc):

        if direc == "forward":
            Si_moments = self._f_Si_moments
            dts = self.forward_dts
        elif direc == "backward":
            Si_moments = self._b_Si_moments
            dts = self.backward_dts

        res = np.diag(np.ones(self._K))
        for ind in range(i+1, n):
            Sind_mean = Si_moments[ind][0]
            res = np.dot(Sind_mean.reshape(self._K, self._K), res)

        return res

    ###
    # For the variable
    #
    #     P_{i,n} = S(ψ_{n-1}) ... S(ψ_{i+1})
    #
    # returns E[P_{i, n}^T Q P_{i, n}] based on the current moments of
    # the variational parameters ψ
    #
    # note the transpose in the ordering
    ###
    def _Exp_Pin_mat_quad(self, i, n, Q, direc):

        if direc == "forward":
            Si_moments = self._f_Si_moments
        elif direc == "backward":
            Si_moments = self._b_Si_moments
        
        res = Q
        for m,c in reversed(Si_moments[i+1:n]):
            m, c = _get_mat_T_moments(m, c)
            res = _Exp_mat_quad(m.reshape(self._K, self._K),
                                c, res)
        return res


    def _integrate(self, x0, psis, direc="forward"):
        if direc == "forward":
            _psis = np.column_stack((self.forward_dts, psis))
            ft = self.forward_full_ts
        else:
            _psis = np.column_stack((self.backward_dts, psis))
            ft = self.backward_full_ts

        X = [x0]
        I = np.diag(np.ones(x0.size))
        for psi in _psis:
            S = I + sum(a*p for a, p in zip(self.A, psi))
            X.append(np.dot(S, X[-1]))

        return ft, np.array(X)




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
Matrix random variable utility functions
"""

###
# E[ X M X.T ]
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



##
# if X is a matrix random variable such that
#
# E[vec(X)] = vec_mean,    Cov[vec(X)] = vec_cov
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
