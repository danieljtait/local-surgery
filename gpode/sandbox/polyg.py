import numpy as np
from scipy.integrate import quad
from scipy.misc import factorial, factorial2
from scipy.special import iv, kv
import matplotlib.pyplot as plt


alpha = np.random.uniform()
beta = np.random.normal()

if beta < 0:
    b = 1j*np.sqrt(abs(beta))
else:
    b = np.sqrt(beta)


def integrand(x):
    return np.exp(-alpha*x**4 - beta*x**2)

print("beta:",beta)
print(quad(integrand, -np.inf, np.inf)[0])

def nconst(gamma):
    expr1 = 0.5*np.exp(1/(8*gamma))/np.sqrt(gamma)
    expr2 = kv(0.25, 1/(8*gamma))
    return expr1*expr2

def nconst2(gamma):
    expr1 = 0.5*np.pi/np.sqrt(2)
    expr2 = np.exp(1/(8*gamma))/np.sqrt(gamma)
    expr3 = iv(-0.25, 1/(8*gamma)) + iv(0.25, 1/(8*gamma))
    return np.real(expr1*expr2*expr3)

if beta > 0:
    gamma = alpha/(b**4)    
    nc = nconst(gamma)/np.sqrt(beta)
    print(0.5*np.sqrt(beta/alpha)*np.exp(beta**2/(8*alpha))*kv(0.25, beta**2/(8*alpha)))
else:
    gamma = alpha/(b**4)
    nc = nconst2(gamma)/np.sqrt(abs(beta))
print(nc)    

"""
c = b**2*a

sdsq = -0.5*b

def upx(x, e=0):
    return np.exp(-a*x**4 - x**2 + e*x)

def upy(y):
    return np.exp(-0.5*a*y**2 + b*y)/np.sqrt(y)

nconst, err = quad(upx, -np.inf, np.inf)

def p(x, e):
    return upx(x, e)/nconst

for n in range(10):
    I, err = quad(lambda x: p(x, 0)*x**n, -np.inf, np.inf)
    print("n: {} | I: {}".format(n, I))

def F(e):
    return quad(lambda x: p(x, e), -np.inf, np.inf)[0]

def TF(e, N):
    res = 0.
    for n in range(N+1):
        if n % 2 == 0:
            I = quad(lambda x: x**n*p(x, 0), -np.inf, np.inf)[0]
            res += I*e**n/(factorial(n))
    return res
    
ee = np.linspace(-6., 6., 50)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ee, [F(e) for e in ee], '-.')
for N in [2, 4, 6, 8]:
    ax.plot(ee, [TF(e, N) for e in ee], '+')
plt.show()
"""
