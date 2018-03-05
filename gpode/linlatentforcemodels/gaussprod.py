import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

s0 = 1.
m0 = .5

def px(x):
    return norm.pdf(x, m0, s0)

def pxy(x, y, rho):
    cov = np.array([[s0**2, rho*s0**2],
                    [rho*s0**2, s0**2]])
    return multivariate_normal.pdf([x, y], [m0, m0], cov)

def var_update(r, y, vary, Exj, Varxj):
    mprior = m0 + r*(Exj-m0)
    vprior = (1-r**2)*s0**2

    mdata = y*Exj/(Exj**2 + Varxj)
    vdata = vary/(Exj**2 + Varxj)
    var = 1/(1/vprior + 1/vdata)
    m = var*(mprior/vprior + mdata/vdata)
    return m, var

def var_par(r, y, sdy):
    m1 = 0.1
    v1 = 1.
    mcur = m1
    while True:
        m2, v2 = var_update(r, y, sdy*sdy, m1, v1)
        m1, v1 = var_update(r, y, sdy*sdy, m2, v2)
        delta = m1 - mcur
        if abs(delta) <= 1e-5:
            break
        else:
            mcur = m1
    return m1, v1
    
y = 1.
sdy = 1.

def up1(x, y):
    return norm.pdf(y, loc=x**2, scale=sdy)*px(x)

def up1_logpdf(x, y):
    return norm.logpdf(y, loc=x**2, scale=sdy) + norm.logpdf(x, m0, s0)

def up2(x1, x2, y, r):
    return norm.pdf(y, loc=x1*x2, scale=sdy)*pxy(x1, x2, r)

C1, err1 = quad(lambda x: up1(x, y), -np.inf, np.inf)

def Dkl(mu, sd):
    return quad(lambda x: norm.pdf(x, mu, sd)*(norm.logpdf(x, mu, sd) - up1_logpdf(x, y)),
                -np.inf, np.inf)[0]

res = minimize(lambda x: Dkl(x[0], np.exp(x[1])), [0., 0.])

r = .6
xx = np.linspace(-1.5, 4.5, 121)

fig = plt.figure()
ax = fig.add_subplot(111)

m1 = 0.1
v1 = 1.
mcur = m1
while True:
    m2, v2 = var_update(r, y, sdy*sdy, m1, v1)
    m1, v1 = var_update(r, y, sdy*sdy, m2, v2)
    delta = m1 - mcur
    if abs(delta) <= 1e-5:
        break
    else:
        mcur = m1
print(m1, m2, res.x[0])
ax.plot(xx, up1(xx, y)/C1, 'k-.')
ax.plot(xx, norm.pdf(xx, m1, np.sqrt(v1)), 'r-')
ax.plot(xx, norm.pdf(xx, res.x[0], np.exp(res.x[1])), 'b-.')

print("============")
print(res.fun < Dkl(m1, np.sqrt(v1)))

fig2 = plt.figure()
ax = fig2.add_subplot(111)

ax.plot(res.x[0], np.exp(res.x[1]), 'o')

pars = np.array([var_par(r, y, sdy)
                 for r in [0.1, 0.9, 0.95]])


ax.plot(pars[:, 0], pars[:, 1], '+-')
ax.plot(pars[-1,0], pars[-1, 1], 'rs')

plt.show()



