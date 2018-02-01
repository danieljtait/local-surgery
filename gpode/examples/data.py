

class DataLoader:

    @classmethod
    def load(cls, example, seed, times, noise_scales, *args, **kwargs):
        if example == "bessel jn":
            return bessel_jn_data(seed, times, noise_scales, **kwargs)
        elif example == "mathieu":
            return mathieu_data(seed, times, noise_scales, **kwargs)
        else:
            raise NotImplementedError


def bessel_jn_data(seed,
                   times,
                   noise_scales,
                   order=1):
    from scipy.special import jn
    from scipy.misc import derivative as deriv
    import numpy as np
    np.random.seed(seed)

    A0 = np.array([[0., 1.0],
                   [0., 0.0]])

    A1 = np.array([[0., 0.0],
                   [1., 0.0]])

    A2 = np.array([[0.0, 0.0],
                   [0.0, 1.0]])

    def g1(t):
        return order**2/t**2 - 1

    def g2(t):
        return -1./t

    X = np.column_stack((
        jn(order, times),
        deriv(lambda z: jn(order, z), x0=times, dx=1e-6)
        ))

    Y = X.copy()
    for k, s in enumerate(noise_scales):
        Y[:, k] += np.random.normal(scale=s, size=times.size)

    res_obj = {"time": times,
               "Y": Y,
               "X": X,
               "As": [A0, A1, A2],
               "Gs": [g1(times), g2(times)]}

    return res_obj


def mathieu_data(seed,
                 times,
                 noise_scales,
                 a=1,
                 h=1):
    import numpy as np
    from scipy.integrate import odeint

    A0 = np.array([[0.0, 1.0],
                   [-a, 0.0]])

    A1 = np.array([[0.0, 0.0],
                   [1.0, 0.0]])

    def g1(t):
        return 2*h**2*np.cos(2*t)

    def dXdt(X, t):
        return np.dot(A0 + A1*g1(t), X)

    x0 = [1., 0.]

    X = [x0]

    for ta, tb in zip(times[:-1], times[1:]):
        sol = odeint(dXdt, X[-1], np.linspace(ta, tb, 25))
        X.append(sol[-1])
    X = np.array(X)
    Y = X.copy()

    for k, s in enumerate(noise_scales):
        Y[:, k] += np.random.normal(scale=s, size=times.size)

    res_obj = {"time": times,
               "Y": Y,
               "X": X,
               "As": [A0, A1],
               "Gs": [g1(times)]}

    return res_obj
