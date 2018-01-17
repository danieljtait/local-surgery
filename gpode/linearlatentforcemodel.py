import numpy as np
from scipy.special import erf
#from .kernels import CollectionKernel


ROOTPI = np.sqrt(np.pi)


def hqp(s, t, Dp, Dq, l, nu):
    expr1 = np.exp(Dq*t)*(erf((s-t)/l) + erf(t/l+nu))
    expr2 = np.exp(-Dp*t)*(erf(s/l - nu) + erf(nu))
    return np.exp(nu**2 - Dq*s)*(expr1 - expr2)/(Dp + Dq)


def kfunc(s, t, ind1=0, ind2=0, **kwargs):
    S = kwargs['S']
    D = kwargs['D']
    lScales = kwargs['lScales']
    p = ind1
    q = ind2
    res = np.array([0.5*ROOTPI*S[p,r]*S[q,r]*l*(hqp(s, t, D[p], D[q], l, 0.5*l*D[q])
                                                + hqp(t, s, D[q], D[p], l, 0.5*l*D[p]))
                    for r, l in enumerate(lScales)])
    
    return np.sum(res, axis=0)


def llfmSquareExponKernel(size):
    S = np.diag(np.ones(size))
    D = np.ones(size)
    lScales = np.ones(size)
    return CollectionKernel(kfunc, size, {'lScales': lScales, 'S': S, 'D': D})


######################################################
# Indefinite integral
#   /
#  |
#  | exp(-Dp*s - Dq*t)*exp(-theta*(s-t)**2) ds      #
#  /
#####################################################
def indefinite_integral(s, t, theta, Dp, Dq):
    expr1 = np.sqrt(np.pi)*np.exp(0.25*Dp**2/theta - (Dp+Dq)*t)
    expr2 = erf( (Dp+2*theta*(s-t)) / (2*np.sqrt(theta)) )
    return expr1*expr2 / (2*np.sqrt(theta))

######################################################
# Indefinite integral of
#  /
# |
# | exp(-Dq*t)*exp(-theta*(s-t)**2) dt
# /
###
def needs_a_name(s, t, theta, Dq):
    expr1 = np.sqrt(np.pi)*np.exp(0.25*Dq**2/theta - Dq*s)
    expr2 = erf( (Dq+2*theta*(t-s)) / (2*np.sqrt(theta)) )
    return expr1*expr2 / (2*np.sqrt(theta))    

###
# indefinite integral 
# /
# |
# | exp(a*u)erf(b*u + c) du 
# /
def integral_explinarg_erf(u, a, b, c):
    expr1 = np.exp(a*u)*erf(b*u + c)
    expr2 = np.exp( (a*(a-4*b*c))/(4*b**2) )*erf( (a-2*b*(b*u + c)) / (2*b) )
    return (expr1 + expr2) / a

def definite_integral_explinarg_erf(u1, u0, a, b, c):
    return integral_explinarg_erf(u1, a, b, c) - integral_explinarg_erf(u0, a, b, c)

##
# Definite integral
def definite_integral(s, t, s0, t0, theta, Dp, Dq):
    rootTheta = np.sqrt(theta)
    
    a1 = -(Dp+Dq)
    b1 = -rootTheta
    c1 = 0.5*Dp/rootTheta + rootTheta*s

    a2 = a1
    b2 = b1
    c2 = 0.5*Dp/rootTheta + rootTheta*s0

    I1_ = definite_integral_explinarg_erf(t, t0, a1, b1, c1)
    I2_ = definite_integral_explinarg_erf(t, t0, a2, b2, c2)
    
    I1 = integral_explinarg_erf(t, a1, b1, c1) - integral_explinarg_erf(t0, a1, b1, c1)
    I2 = integral_explinarg_erf(t, a2, b2, c2) - integral_explinarg_erf(t0, a2, b2, c2)

    C = ROOTPI*np.exp(0.25*Dp**2 / theta) / (2*rootTheta) 
    
    return C*(I1 - I2)
    

def kfunc2(s, t, p, q, **kwargs):
    S = kwargs['S']
    D = kwargs['D']
    invsqlScales = kwargs["theta1s"]

    def C(r):
        return S[p, r]*S[q, r]
    
    res = np.array([C(r)*definite_integral(s, t, 0., 0., theta, D[p], D[q])
                    for r, theta in enumerate(invsqlScales)])

    return np.exp(D[p]*s+D[q]*t)*np.sum(res, axis=0)

def kfunc3(s, t, p, q, **kwargs):
    thetas = kwargs["thetas"]
    S = kwargs["S"]
    D = kwargs["D"]

    outputDim = len(D)

    def C(r):
        return S[p, r]*S[q, r]

    if p < outputDim and q < outputDim:
        # Dealing with covar {X_p(s), X_q(t) }
        res = np.array([C(r)*definite_integral(s, t, 0., 0., theta, D[p], D[q])
                        for r, theta in enumerate(thetas)])
        return np.exp(D[p]*s + D[q]*t)*np.sum(res, axis=0)
    elif p < outputDim and q >= outputDim:
        # Dealing with covar {X_p(s), f_{q % dim}(t) }
        r = q - outputDim
        res = needs_a_name(t, s, thetas[r], D[p]) - needs_a_name(t, 0, thetas[r], D[p])
        return S[p, r]*np.exp(D[p]*s)*res
        
    elif p >= outputDim and q < outputDim:
        # Dealing with covar {f_{p % t}(s), X_q(t) }
        pass
    else:
        pass

def collection_kernel_kfunc(s, t, ind1, ind2, **kwargs):
    pass

"""
S = np.array([[0.3, 0.1],
              [0.0, 0.2]])
D = np.array([1.1, 0.5])
thetas = np.array([1.1, 2.])
ss = np.linspace(1.1, 1.5, 3)
tt = np.linspace(0.5, 2.1, 3)

val = 0.

s = ss[1]
t = tt[1]

p = 1
q = 1
print(kfunc2(ss, tt, p, q, S=S, D=D, theta1s = thetas))
from scipy.integrate import dblquad
for r, l in enumerate(thetas):
    def integrand(y, x):
        c = np.exp(D[p]*(s-x) + D[q]*(t-y))
        c *= S[p, r]*S[q, r]
        return c*np.exp(-l*(x-y)**2)

    I, err = dblquad(integrand, 0., s, lambda x: 0., lambda x: t)
    val += I

print(val)
"""
