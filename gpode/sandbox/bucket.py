from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, jn
from scipy.misc import derivative as deriv

a = 2
def A(t):
    return np.array([[0., 1.],
                     [(a**2-t**2)/t**2, -1/t]])

def dXdt(X, t):
    return np.dot(A(t), X)


eps = 1e-5
tt = np.linspace(eps, 10.)

x0 = [jn(a, eps), deriv(lambda z: jn(a, z), x0=eps, dx=1e-6)]


sol = odeint(dXdt, x0, tt)

print(x0)

def bessel_data(order, tt):
    y = jn(order, tt)
    dy = deriv(lambda z: jn(order, z), x0=tt, dx=1e-6)
    return np.column_stack((y, dy))

times = np.linspace(0.5, 5., 5)

Y = bessel_data(a, times)
print(Y)

plt.plot(tt, sol, 'k-.', alpha=0.2)
plt.plot(tt, jn(a, tt), '+')
plt.plot(times, Y, 'o')

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot(times, -1/times)
#ax.plot(times, (a**2-times**2)/times**2)

plt.show()
