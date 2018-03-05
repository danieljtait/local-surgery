import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import iv, kv, gamma, hyp1f1, factorial
import matplotlib.pyplot as plt

##
# Returns the integral of the function
#
# exp(-x^4 + b*x^2)
def _canonical_normalising_const(b):
    arg = b**2/8
    denom = 2*np.sqrt(2*abs(b))
    return np.pi*np.exp(arg)*(abs(b)*iv(-0.25, arg) + b*iv(0.25, arg))/denom

##
# Returns the integral of the function
#
# exp(-a*x**4 + b*x**2)
def _normalising_const(a, b):
    return _canonical_normalising_const(b/np.sqrt(a))*np.power(a, -0.25)


def _Z1ba(b, a, order=10):
    order = 25
    ks = np.arange(0, order + 1)
    J, N = np.meshgrid(ks, ks)

    def _func(j, n):
        expr1 = a**(2*j)/factorial(2*j)        
        expr2 = b**n/factorial(n)
        expr3 = gamma(0.5*(j+n) + 0.25)
        return expr1*expr2*expr3

    return 0.5*np.sum(_func(J.ravel(), N.ravel()))

def _Z1ba2(b, a, order=10):
    js = np.arange(0, order+1)

    _arg = 0.25*(2*js + 1)
    
    expr1 = gamma(_arg)*hyp1f1(_arg, 0.5, 0.25*b**2)
    expr2 = b*gamma(_arg+0.5)*hyp1f1(_arg+0.5, 1.5, 0.25*b**2)
    expr3 = a**(2*js)/factorial(2*js)
    return 0.5*sum((expr1 + expr2)*expr3)

def _Zcba(c, b, a):
    return c*_Z1ba(b*c**2, a*c)

a = np.random.normal()
b = np.random.normal()
c = 1.
print(c, b, a)
xx = np.linspace(-1.5, 1.5, 150)

Z = _Zcba(c, b, a)
print(quad(lambda x: x*np.exp(-(x/c)**4 + b*x**2 + a*x)/Z, -np.inf, np.inf))


plt.plot(xx, np.exp(-(xx/c)**4 + b*xx**2 + a*xx)/_Zcba(c, b, a))

order = 25
ks = np.arange(0, order+1)
J, N = np.meshgrid(ks[1:], ks)
def _func(j, n):
    expr1 = a**(2*j-1)/factorial(2*j-1)
    expr2 = b**n/factorial(n)
    expr3 = gamma(0.5*(j+n) + 0.25)
    return 0.5*expr1*expr2*expr3

Ex_num = np.sum(_func(J.ravel(), N.ravel()))

def integrand(t, j, b):
    return np.power(t, j/2 + 1/4 - 1)*np.exp(b*np.sqrt(t) - t)


a =  10.52
b = 5.52

class quartic_expon:
    def __init__(self, eta1=1, eta2=0, eta3=0):
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3

        eta1_4root = eta1**0.25
        self._scaled_eta2 = eta2/(eta1_4root**2)
        self._scaled_eta3 = eta3/(eta1_4root)

        self._norm_const = _Z1ba2(self._scaled_eta2, self._scaled_eta3)/eta1_4root

        self.EX = quad(lambda x: x*self.pdf(x), -np.inf, np.inf)[0]
        self.EX2 = quad(lambda x: x**2*self.pdf(x), -np.inf, np.inf)[0]
        self.EX4 = quad(lambda x: x**4*self.pdf(x), -np.inf, np.inf)[0]

    def _unnorm_pdf(self, x):
        arg = -self.eta1*x**4 + self.eta2*x**2 + self.eta3*x
        return np.exp(arg)

    def pdf(self, x):
        return self._unnorm_pdf(x)/self._norm_const

    def logpdf(self, x):
        arg = -self.eta1*x**4 + self.eta2*x**2 + self.eta3*x        
        return arg - np.log(self._norm_const)

p = quartic_expon(1.5, -1., 0.)

def var_update(moments, A4, A2, A1):
    EX = moments[:, 0]
    EX2 = moments[:, 1]
    EX4 = moments[:, 2]
    
    _pars = []
    
    for k, m in enumerate(moments):
        eta1 = A4[k, k]
        eta2 = 2*(np.dot(A4[k, :], EX2) - A4[k, k]*EX2[k]) + A2[k, k]
        eta3 = 2*(np.dot(A2[k, :], EX) - A2[k, k]*EX[k]) + A1[k]

        _p = quartic_expon(eta1, eta2, eta3)

        EX[k] = _p.EX
        EX2[k] = _p.EX2
        EX4[k] = _p.EX4

        _pars.append([eta1, eta2, eta3])

    return moments, np.array(_pars)
        
A4 = np.array([[1.1, 0.3],[0.3, 1.1]])
A2 = np.random.normal(size=4).reshape(2, 2)
A2 = 0.5*(A2 + A2.T)
A1 = np.random.normal(size=2)


init_pars = np.array([[1., 0., 0.],
                      [1., 0., 0.]])
init_moms = []
for par in init_pars:
    p = quartic_expon(*par)
    init_moms.append([p.EX, p.EX2, p.EX4])
init_moms = np.array(init_moms)
print(init_moms)

params = np.zeros(init_pars.shape)
mcur = params.copy()
while True:
    moms, pars = var_update(init_moms, A4, A2, A1)
    params = pars.copy()

    delta = np.linalg.norm(mcur.ravel() - pars.ravel())
    if delta <= 1e-8:
        break
    mcur = pars.copy()

def _integrand(y, x):
    z = np.array([x, y])
    expr1 = -np.dot(z**2, np.dot(A4, z**2))
    expr2 = np.dot(z, np.dot(A2, z))
    expr3 = np.dot(A1, z)
    return np.exp(expr1 + expr2 + expr3)

Ztrue = dblquad(_integrand, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]

print(dblquad(lambda y, x: x*_integrand(y, x)/Ztrue,
              -np.inf, np.inf,
              lambda x: -np.inf, lambda x: np.inf))

print(dblquad(lambda y, x: y*_integrand(y, x)/Ztrue,
              -np.inf, np.inf,
              lambda x: -np.inf, lambda x: np.inf))
print("---A2---")
print(A2)
print("---A1---")
print(A1)
p1 = quartic_expon(*params[0, :])
p2 = quartic_expon(*params[1, :])
print(p1.EX)
print(p2.EX)
#print(p1._norm_const*p2._norm_const)

xx1 = np.linspace(-2., 2., 50)
xx2 = np.linspace(-2., 2., 50)
X2, X1 = np.meshgrid(xx2, xx1)

P1 = np.zeros((xx1.size, xx2.size))
for i, x1 in enumerate(xx1):
    for j, x2 in enumerate(xx2):
        P1[i, j] = _integrand(x2, x1)/Ztrue
P2 = p1.pdf(X1.ravel())*p2.pdf(X2.ravel())
P2 = P2.reshape(X2.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(xx1, xx2, P1)
ax.contour(xx1, xx2, P2)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(xx1, p1.pdf(xx1), 'b-')
ax.plot(xx2, p2.pdf(xx2), 'r-')


b = 2
p = quartic_expon(1, b, 0)

xx = np.linspace(-3., 3., 250)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xx, p.pdf(xx), '-.')

r = np.sqrt(b/2)
from scipy.stats import norm

sd = 1/np.sqrt(4*b)

def qmix(x):
    val1 = 0.5*norm.pdf(xx, loc=-r, scale=sd)
    val2 = 0.5*norm.pdf(xx, loc=r, scale=sd)
    return val1 + val2

ax.plot(xx, qmix(xx))

plt.show()
    
"""
b = 0.5
from scipy.stats import norm, multivariate_normal
from scipy.integrate import dblquad
print("==============")
print(quad(lambda x: np.exp(-x**4 + b*x**2), -np.inf, np.inf))
for sd0 in [1., 2., 5.]:
    C = np.sqrt(2*np.pi*sd0**2)
    print(quad(lambda x: C*np.exp(-x**4 + b*x**2)*norm.pdf(x, scale=sd0), -np.inf, np.inf))
print("==============")
def integrand(y, x, sd0, r):
    cov = sd0**2*np.array([[1., r], [r, 1.]])
    return np.exp(-x**2*y**2 + b*x*y)*multivariate_normal.pdf([x, y], mean=np.zeros(2), cov=cov)
print(quad(lambda x: np.exp(-x**4 + b*x**2)*norm.pdf(x, scale=sd0), -np.inf, np.inf))
print("---------")
for r in [0.9, 0.95, 0.99, 0.999, 0.9999]:
    print(dblquad(integrand, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf, args=(sd0, r)))
"""
#plt.show()
