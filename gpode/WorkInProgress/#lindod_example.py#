import numpy as np
import gpode.kernels
import gpode.gaussianprocesses
from scipy.integrate import quad, odeint
from scipy.stats import norm, multivariate_normal
from lindod_src import (MGPLinearAdapGrad, Data, state_update,
                        log_updf, gr_post_conditional,
                        xk_post_conditional)
import matplotlib.pyplot as plt
from scipy.optimize import minimize


A0 = np.array([[0.0, 0.],
               [0.0, 0.]])
A1 = np.array([[0.0, -1.],
               [1.0, 0.]])

x0 = np.array([1., 0.])
tt = np.linspace(0., 3., 8)
Y = odeint(lambda x, t: np.dot(A0 + np.cos(t)*A1, x), x0, tt)

data = Data(tt, Y)

X_gps = [gpode.gaussianprocesses.GradientGaussianProcess(
    gpode.kernels.GradientMultioutputKernel.SquareExponKernel([10., 5.]))
         for k in range(2)]

mod = MGPLinearAdapGrad([A0, A1], data, X_gps, latent_force_vals=[np.cos(tt)])
mod.init()

As = mod.As
Gs = mod.latent_force_vals
X = mod.X

print(mod._dXdt(X, As, Gs))

k = 1
r = 0
vkr = np.array([ar_kj*xj for ar_kj, xj in zip(As[r][k,:], X.T)]).T
#print(As[r][k,0]*X[:, 0])
Vk = [np.sum(np.array([ar_kj*xj for ar_kj, xj in zip(As[r][k, :], X.T)]).T, axis=1)
              for r, g in enumerate(Gs)]

print("==============")
print("Hello, world!")
print("==============")
#print(np.sum(Vk, axis=0))


###
# Calculates the mean and variance for component k
# of the product of Gaussians giving the posterior conditional of 
def afunc(r, k, As, Gs):
    R = len(As)

    Vk = [np.sum(np.array([ar_kj*xj
                           for ar_kj, xj in zip(As[_r][k, :], X.T)]), axis=0)
          for _r in range(R)]

    a = Vk[r]
    b = sum(g*Vk[s] for s, g in enumerate(Gs) if s != r)

    a_ = np.diag(a)
    print(np.linalg.pinv(a_))

    # 
    #    for v in Vk:
#        a =





LCs = [gp.Ls["y"] for gp in mod.latent_gps]
dms = []
dCs = []

for gp in mod.latent_gps:
    dm, dc = gp.predict(dy_inputs=gp.training_data["y"][0],
                        ret_par=True)
    dms.append(dm)
    dCs.append(dc)



def pk(gr, r, k, As, Gs, dms, dCs, X):
    R = len(As)

    Vk = [np.sum(np.array([ar_kj*xj
                           for ar_kj, xj in zip(As[_r][k, :], X.T)]), axis=0)
          for _r in range(R)]

    ak = Vk[r]
    bk = dms[k] - np.sum((vks for s, vks in enumerate(Vk) if s != r), axis=0)


    cov_inv = np.linalg.inv(dCs[k])
    Dak = np.diag(ak)

    Lam = np.dot(Dak, np.dot(cov_inv, Dak))
    mean = np.dot(np.linalg.inv(Dak), bk)

    return mean, Lam


def gr_cond_moments(r, As, Gs, dms, dCs, X):
    mk = []
    Ckinv = []
    for k in range(X.shape[1]):
        m, cinv = pk(None, r, k, As, Gs, dms, dCs, X)
        mk.append(m)
        Ckinv.append(cinv)
    Sigma_inv = Ckinv[0] + Ckinv[1]
    Sigma = np.linalg.inv(Sigma_inv)
    mean = np.dot(Ckinv[0], mk[0]) + np.dot(Ckinv[1], mk[1])
    mean = np.dot(Sigma, mean)
    return mean, Sigma


def qk(gr, r, k, As, Gs, dms, dCs, X):
    Gs_ = [g for g in Gs]
    Gs_[r] = gr

    F = mod._dXdt(X, As, Gs_)

    etak = F[:, k] - dms[k]

    cov_inv = np.linalg.inv(dCs[k])
    exp_arg = -.5*np.dot(etak, np.dot(cov_inv, etak))
    return exp_arg

r = 1
k = 0
z = np.random.normal(size=tt.size)
m, _ = pk(z, r, k, As, Gs, dms, dCs, mod.X)
res = minimize(lambda z: -qk(z, r, k, As, Gs, dms, dCs, mod.X),
               z)
print(res.message)
print("res.fun", res.fun)
fm = -qk(m, r, k, As, Gs, dms, dCs, mod.X)
print("fun(m)", fm)
print("f(m) < f(res.x)?", fm < res.fun)

print("-------")
def objfunc(g):
    val = 0.
    for k in range(2):
        val += qk(g, 1, k, As, Gs, dms, dCs, mod.X)
    return -val
res = minimize(objfunc, z)
print(res.x)
print(np.diag(res.hess_inv))
m, S = gr_cond_moments(1, As, Gs, dms, dCs, X)
print(m)
print(np.diag(S))

s_, t_ = np.meshgrid(tt, tt)
C0 = np.exp(-1.1*(t_.ravel()-s_.ravel())**2).reshape(s_.shape)

post_cov = np.linalg.inv(np.linalg.inv(C0) + np.linalg.inv(S))
post_mean = np.dot(post_cov, np.dot(np.linalg.inv(S), m))
print("Posterior mean")
print(post_mean)
print("Posterior cond var")
print(np.diag(post_cov))
#print(res.x)
#print(res.hess_inv)

prior = multivariate_normal(mean=np.zeros(tt.size), cov=C0)
m = gr_post_conditional(1, As, Gs,
                        [0.1, 0.1], [0.05, 0.05],
                        mod.X, dms, LCs, dCs,
                        mod._dXdt, prior)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt, m, '+')
ax.plot(tt, np.cos(tt), 's')

Cx = np.exp(-3*(s_.ravel()-t_.ravel())**2).reshape(s_.shape)
xprior = multivariate_normal(mean=np.zeros(tt.size),
                             cov=Cx)



fig = plt.figure()
ax = fig.add_subplot(111)


for nt in range(100):
    for k in range(2):
        mxk, C_xk = xk_post_conditional(k, As, Gs,
                                        [0.01, 0.01], [0.05, 0.05],
                                        mod.data.Y, mod.X,
                                        dms, LCs, dCs,
                                        mod._dXdt, xprior)
        rxk = multivariate_normal.rvs(mean=mxk, cov=C_xk)
        mod.X[:, k] = rxk
    ax.plot(tt, mod.X[:, 0], 'k+', alpha=0.2)
ax.plot(tt, Y[:, 0], 's')
plt.show()
    
"""
S, T = np.meshgrid(tt, tt)
Sigma = np.exp(-0.5*(S.ravel()-T.ravel())**2).reshape(S.shape)


def objfunc(g):
    Gs = [np.ones(tt.size), g]
    lp = log_updf(As, Gs,
                  [0.1, 0.1], [0.05, 0.05],
                  X, dms, LCs, dCs, mod._dXdt)
    lprior = 0.  # multivariate_normal.logpdf(g, np.zeros(g.size), cov=Sigma)
    return -(lp + lprior)


g = np.cos(tt)


def q(gi, i):
    g_ = g.copy()
    g_[i] = gi
    return np.exp(-objfunc(g_))


def gr_cond_par(r, X, As, Sigmas):
    K = X.shape[1]

    m = []
    C = []
    for k in range(K):
        brk = np.sum([ar_kj*xj for ar_kj, xj in zip(As[r][k, :], X.T)], axis=0)
        Brk = np.diag(brk)

        sigmas_inv = np.linalg.inv(Sigmas[k])
        cov_inv = np.dot(Brk, np.dot(sigmas_inv, Brk))

        C.append(np.linalg.inv(cov_inv))

    return m, C


i = 2
xx = np.linspace(np.cos(tt[i])-0.1, np.cos(tt[i])+0.1)

n_const, err = quad(lambda x: q(x, i), -np.inf, np.inf)
m, merr = quad(lambda x: x*q(x, i)/n_const, -np.inf, np.inf)
v, verr = quad(lambda x: (x-m)**2*q(x, i)/n_const, -np.inf, np.inf)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx, [q(x, i)/n_const for x in xx])
ax.plot(xx, norm.pdf(xx, loc=m, scale=np.sqrt(v)), '+')
plt.show()
"""

#res = minimize(objfunc, np.zeros(tt.size))
#print(res)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(tt, np.cos(tt))
#ax.plot(tt, res.x, '+')
#plt.show()
