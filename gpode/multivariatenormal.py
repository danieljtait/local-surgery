import numpy as np

def cholesky_decomp_helper(C, delta=1e-6, count=0):
    try:
        return np.linalg.cholesky(C)
    except:
        MAX_COUNT = 10
        nt = 0
        delta = 1e-4
        I = np.diag(np.ones(C.shape[0]))
        while nt < MAX_COUNT:
            try:
                return np.linalg.cholesky(C + delta*I)
            except:
                delta *= 2
            nt += 1

        # Raise exception if we get here
        

#################################################
# If x1, x2 are jointly normal distributed then
# returns the parameters of the distribution
#
#     p(x1 | x2) ~ N(condMean, condCov)
#
def CondPar(x2, m1, m2, C11, C12, C22):
    L = np.linalg.cholesky(C22)
    condMean = m1 + np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, x2-m2)))
    condCov = C11 - np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, C12.T)))
    return condMean, condCov


####
# Simulates a single sample from the multivariate
# normal distribution
def Sim(mean, cov=None, L=None):
    # Looking for the Cholesky decomposition to have been passed
    # as an argument
    if L:
        return mean + np.dot(L, np.random.normal(size=L.shape[0]))
    else:
        L = np.linalg.cholesky(cov)
        return Sim(mean, L=L)

##
# Gradient of the multivariate normal log pdf
# with respect to the mean m and covariance C
def parGrad(x, m, C, par="both"):
    L = cholesky_decomp_helper(C)
    if par == "mean":
        return np.linalg.solve(L.T, np.linalg.solve(L, x-m))
    elif par == "cov":
        Cinv = np.linalg.solve(L.T, np.linalg.solve(L, np.diag(np.ones(C.shape[0]))))
        return -0.5*(Cinv - np.dot(Cinv, np.dot(np.outer(x-m, x-m), Cinv)))
    else:
        dm = np.linalg.solve(L.T, np.linalg.solve(L, x-m))
        Cinv = np.linalg.solve(L.T, np.linalg.solve(L, np.diag(np.ones(C.shape[0]))))
        dC = -0.5*(Cinv - np.dot(Cinv, np.dot(np.outer(x-m, x-m), Cinv)))
        return dm, dC

####
# gradient of mean zero GP fitted to data Y at time points tt is given by
# with respect to parameter p is
#
#  sum( [ dLdC ] * [ dkdp ] )
# 

"""

from scipy.stats import multivariate_normal

ss = np.linspace(0., 1., 3)
T, S = np.meshgrid(ss, ss)

mean = np.zeros(ss.size)
cov = np.exp(-(S.ravel() - T.ravel())**2).reshape(T.shape)
x = multivariate_normal.rvs(mean=mean, cov=cov)

def lp(x, m, c):
    return multivariate_normal.logpdf(x, mean=m, cov=c)

def f(par):
    cov = par*np.exp(-(S.ravel() - T.ravel())**2).reshape(T.shape)
    return lp(x, mean, cov)

def df(par):
    C = par*np.exp(-(S.ravel() - T.ravel())**2).reshape(T.shape)
    dCdp = np.exp(-(S.ravel() - T.ravel())**2).reshape(T.shape)
    dLdC = parGrad(x, mean, C, par="cov")
    return np.sum(dLdC*dCdp)

def g(theta):
    cov = theta[0]*np.exp(-theta[1]*(S.ravel()-T.ravel())**2).reshape(T.shape)
    return lp(x, mean, cov)

def dfdtheta(par, x, mean, C, S, T):
    dCdtheta0 = C/par[0]
    dCdtheta1 = (-(S.ravel() - T.ravel())**2).reshape(T.shape) * C
    dLdC = parGrad(x, mean, C, par="cov")
    return np.sum(dLdC*dCdtheta0), np.sum(dLdC*dCdtheta1)

eps = 1e-6

theta = [1.1, 0.5]
thetap = [1.1, 0.5+eps]

# Try an example
inputs = np.linspace(0., 15., 21)
Y = np.cos(inputs)

####
T, S = np.meshgrid(inputs, inputs)
Travel = T.ravel()
Sravel = S.ravel()

mean = np.zeros(inputs.size)

def objfunc(theta):
    try:
        cov = theta[0]*np.exp(-theta[1]*(Sravel - Travel)**2).reshape(T.shape)
        retval =-lp(Y, mean, cov)
    except:
        retval = np.inf
    return retval

def objfuncGrad(theta):
    try:
        C0 = np.exp(-theta[1]*(Sravel - Travel)**2).reshape(T.shape)
        cov = theta[0]*C0
        dCdtheta0 = C0
        dCdtheta1 = -((Sravel - Travel)**2).reshape(T.shape) * cov

        dLdC = parGrad(Y, mean, cov, par="cov")
        
        d1 = -np.sum(dLdC*dCdtheta0)
        d2 = -np.sum(dLdC*dCdtheta1)

        return np.array([d1, d2])
    except:
        return np.array([np.inf, np.inf])
        
from scipy.optimize import minimize

p = (1, 1.1)
pp = (1, p[1]+eps)

def covar(p):
    return p[0]*np.exp(-p[1]*(Sravel-Travel)**2).reshape(T.shape)



###
# Check if this problem is a conjugate gradient one

#func = lambda x1, x2 : -dfdtheta((x1, x2), Y, mean, covar((x1, x2)), S, T)
#print((objfunc(pp)-objfunc(p))/eps)
#grad = func(p[0], p[1])
bnds = ((0, None), (0, None))
res0 = minimize(objfunc, (1, 1), method="Nelder-Mead")
res1 = minimize(objfunc, (1, 1), method='SLSQP', jac=objfuncGrad, bounds=bnds)
res2 = minimize(objfunc, (1, 1), method='SLSQP', bounds=bnds)
res3 = minimize(objfunc, (1, 1), method='BFGS')
res4 = minimize(objfunc, (1, 1), jac=objfuncGrad, method="BFGS")
res5 = minimize(objfunc, (1, 1), jac=objfuncGrad, method="CG")

print(res0)
print("")
print(res1)
print("")
print(res2)
print("")
print(res3)
print("")
print(res4)
print("")
print(res5)
"""

    


