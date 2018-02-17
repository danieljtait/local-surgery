import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt


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


theta0 = 1.
theta1 = 1.
sa = 0.
sb = .5

N = 5

ss = np.linspace(sa, sb, N)

S2, S1 = np.meshgrid(ss, ss)
cov = theta0*np.exp(-theta1*(S1.ravel()-S2.ravel())**2).reshape(S2.shape)
Lgg = np.linalg.cholesky(cov)

nsim = 1000

rvI = []
rv2 = []

mf_vars = np.diag(cov)    
mf_sds = np.sqrt(mf_vars)

print(np.diag(mf_vars))
print(cov)

for nt in range(nsim):
    z = np.dot(Lgg, np.random.normal(size=N))
    df = z[1:]**2 + z[:-1]**2
    I = sum(0.5*df*np.diff(ss))

    m_mf = np.random.normal(scale=mf_sds)
    df2 = m_mf[1:]**2 + m_mf[:-1]**2
    I2 = sum(0.5*df2*np.diff(ss))
    
    rvI.append(I)
    rv2.append(I2)

j_var = _cov_funcs["J1J1"](sb, sb, sa, sa, theta0, theta1)
Js = np.random.normal(size=nsim, loc=0., scale=np.sqrt(j_var))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(rvI, 33, normed=True)
ax.hist(rv2, 33, normed=True)
#ax.hist(Js**2, 50, normed=True, facecolor='red', alpha=0.2)
ax.set_xlim((0., 4.))
plt.show()



