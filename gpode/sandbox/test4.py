import numpy as np
from gpode.latentforcemodels import VariationalMLFM2
from scipy.integrate import quad, dblquad, odeint
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.set_printoptions(precision=2)

A0 = np.array([[0., 0.], [0., 0.]])
A1 = np.array([[0., -1.], [1., 0.]])
A2 = np.array([[ 0., 0.], [0., 1.]])

g1 = lambda t: np.cos(t)
g2 = lambda t: np.exp(-0.5*(t-2)**2)

x0 = np.array([1., 0.0])
ttDense = np.linspace(0., 3., 50)

sol = odeint(lambda x, t: np.dot(A0 + A1*g1(t) + g2(t)*A2, x), x0, ttDense)
redInd = np.linspace(0, ttDense.size-1, 9, dtype=np.intp)
tt = ttDense[redInd]
yy = sol[redInd, ]

vobj = VariationalMLFM2(1, tt, yy, .1,
                        As=[A0, A1, A2],
                        lscales=[1.0, 1.0],
                        obs_noise_priors=[[120, .5],
                                          [120, .5]])

def fit(vobj):
    x0 = vobj._f_Ex[0]    
    I = np.diag(np.ones(2))

    Yb = vobj.backward_data[1:]
    Yf = vobj.forward_data[1:]
#    Cov = np.diag(vobj.obs_noise_pars[1]/vobj.obs_noise_pars[0])    
    Cov = np.diag(0.1*np.ones(2))
    def _objfunc(psi):
        psi_b = psi[:vobj._N_b_psi*vobj._R].reshape(vobj._N_b_psi, vobj._R)
        psi_f = psi[vobj._N_b_psi*vobj._R:].reshape(vobj._N_f_psi, vobj._R)

        yb = [x0]
        for f in psi_b:
            yb.append(np.dot(I + A1*f[0] + A2*f[1], yb[-1]))
        yb = np.array(yb)
        yb = yb[vobj.backward_data_inds[1:], :]

        yf = [x0]
        for f in psi_f:
            yf.append(np.dot(I + A1*f[0] + A2*f[1], yf[-1]))
        yf = np.array(yf)
        yf = yf[vobj.forward_data_inds[1:], :]

        val1 = 0.
        for i in range(Yb.shape[0]):
            val1 += multivariate_normal.logpdf(Yb[i, ], yb[i, ], cov=Cov)
            
        val2 = 0.
        for i in range(Yf.shape[0]):
            val2 += multivariate_normal.logpdf(Yf[i, ], yf[i, ], cov=Cov)

        return -(val1+val2)

    N = vobj._N_b_psi + vobj._N_f_psi

    f0 = []
    for _ta, _tb in zip(vobj.backward_full_ts[:-1], vobj.backward_full_ts[1:]):
        J1 = quad(g1, _ta, _tb)[0]
        J2 = quad(g2, _ta, _tb)[0]
        f0.append([J1, J2])
    for _ta, _tb in zip(vobj.forward_full_ts[:-1], vobj.forward_full_ts[1:]):
        J1 = quad(g1, _ta, _tb)[0]
        J2 = quad(g2, _ta, _tb)[0]
        f0.append([J1, J2])
    f0 = np.array(f0)
    f0 = f0.ravel()

    res = minimize(_objfunc, f0, method="Nelder-Mead")
    print(res.message)
    res = minimize(_objfunc, res.x, method="Nelder-Mead")
    print(res.message)    

    psi_b = res.x[:vobj._N_b_psi*vobj._R].reshape(vobj._N_b_psi, vobj._R)
    psi_f = res.x[vobj._N_b_psi*vobj._R:].reshape(vobj._N_f_psi, vobj._R)
    return psi_b, psi_f
    

fig1 = plt.figure()
fig2 = plt.figure()
#fig3 = plt.figure()
#fig4 = plt.figure()

ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
#ax3 = fig3.add_subplot(111)
#ax4 = fig4.add_subplot(111)
ss = np.linspace(0., 3., 100)

ax1.plot(ttDense, sol[:, 0], alpha=0.5)
ax2.plot(ttDense, sol[:, 1], alpha=0.5)

vobj._gp_scale_alphas = np.array([2., 2.])
vobj._gp_scale_betas = np.array([50.0, 50.0])

pcur = vobj._gp_scale_alphas/vobj._gp_scale_betas
psi_cur = np.row_stack((np.array(vobj._forward_psi.mean),
                        np.array(vobj._backward_psi.mean)))

for nt in range(50):

    
    for i in range(vobj._N_b_psi):
        vobj._update_psi_i(i, "backward")

    for i in range(vobj._N_f_psi):
        vobj._update_psi_i(i, "forward")

#    if nt > 25:
#        vobj._update_gp_scale_pars()

    vobj._update_noise_pars()

    pnew = vobj._gp_scale_alphas/vobj._gp_scale_betas


    psi_new = np.row_stack((np.array(vobj._forward_psi.mean),
                            np.array(vobj._backward_psi.mean)))
    
    delta = np.linalg.norm(psi_cur-psi_new)
    print("Expected gp scales:", vobj._gp_scale_betas/(vobj._gp_scale_alphas-1))
    print(vobj.obs_noise_pars[1]/(vobj.obs_noise_pars[0]-1))
#    print(np.array(vobj._forward_psi.mean))

    
    if delta < 1e-3 and nt > 20:
        break
    pcur = pnew

    Exf = np.array(vobj._f_Ex)
    Exb = np.array(vobj._b_Ex)

    ax1.plot(vobj.forward_full_ts, Exf[:, 0], 'b-', alpha=0.2)
    ax2.plot(vobj.forward_full_ts, Exf[:, 1], 'r-', alpha=0.2)
    
    ax1.plot(vobj.backward_full_ts, Exb[:, 0], 'b-', alpha=0.2)
    ax2.plot(vobj.backward_full_ts, Exb[:, 1], 'r-', alpha=0.2)    


    psi_pred = vobj.pred_latent_force(0, ss,
                                      np.array(vobj._backward_psi.mean)[:, 0],
                                      np.array(vobj._forward_psi.mean)[:, 0])

    psi_pred2 = vobj.pred_latent_force(1, ss,
                                       np.array(vobj._backward_psi.mean)[:, 1],
                                       np.array(vobj._forward_psi.mean)[:, 1])

#    if nt > 25:
#        ax3.plot(ss, psi_pred, 'k-', alpha=0.2)
#        ax4.plot(ss, psi_pred2, 'k-', alpha=0.2)

ax1.plot(vobj.forward_full_ts, Exf[:, 0], 'k-o', alpha=0.8)
ax2.plot(vobj.forward_full_ts, Exf[:, 1], 'k-o', alpha=0.8)
    
ax1.plot(vobj.backward_full_ts, Exb[:, 0], 'k-o', alpha=0.8)
ax2.plot(vobj.backward_full_ts, Exb[:, 1], 'k-o', alpha=0.8)    


ax1.plot(vobj.backward_full_ts[vobj.backward_data_inds[1:]],
         vobj.backward_data[1:, 0], 's')
ax1.plot(vobj.forward_full_ts[vobj.forward_data_inds],
         vobj.forward_data[:, 0], 's')

ax2.plot(vobj.backward_full_ts[vobj.backward_data_inds[1:]],
         vobj.backward_data[1:, 1], 's')
ax2.plot(vobj.forward_full_ts[vobj.forward_data_inds],
         vobj.forward_data[:, 1], 's')

fig3 = plt.figure()
ax = fig3.add_subplot(111)

ax.plot(vobj.forward_full_ts[1:],
        np.array(vobj._forward_psi.mean)[:, 0], '+')

plt.show()

