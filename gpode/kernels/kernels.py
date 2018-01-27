import numpy as np
from gpode.bayes import Parameter


##
# Attach a parameter to the kernel allowing for
# specifications of priors, proposals etc.,
# the val_map argument allows for some parameters
# to be kept constant
class KernelParameter(Parameter):
    def __init__(self, val_map, *args, **kwargs):
        super(KernelParameter, self).__init__(*args, **kwargs)
        self.val_map = val_map

    def get_value(self):
        return self.val_map(self.value)


class Kernel:
    def __init__(self, kfunc, kpar):
        self.kfunc = kfunc  # Callable giving cov{Y(x1),Y(x2)}
        self.kpar = kpar    # additional arguments to kfunc

    def cov(self, x1, x2=None, kpar=None):
        if kpar is None:
            kpar = self.kpar

        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                # What is it? - raise exception
                raise ValueError

        elif isinstance(x2, (float, list)):
            x2 = np.asarray(x2)

        T, S = np.meshgrid(x2, x1)
        return self.kfunc(S.ravel(), T.ravel(), kpar).reshape(T.shape)

    @classmethod
    def SquareExponKernel(cls, kpar=None):
        if not isinstance(kpar, np.ndarray):
            if kpar is None:
                kpar = [1., 1.]
        return cls(lambda s, t, p: p[0]*np.exp(-p[1]*(s-t)**2), kpar)


##
# Extends the Kernel class to model the covariance
# between outputs Y_p(t) and Y_q(t), the callable kfunc
# now has additional positional arguments p, q
class MultioutputKernel(Kernel):
    def __init__(self, kfunc, kpar):
        super(MultioutputKernel, self).__init__(kfunc, kpar)

    def cov(self, ind1, ind2, x1, x2=None, kpar=None):
        if kpar is None:
            if isinstance(self.kpar, KernelParameter):
                kpar = self.kpar.get_value()
            else:
                kpar = self.kpar

        if not isinstance(x1, np.ndarray):
            x1 = np.asarray(x1)

        if not isinstance(x2, (float, list, np.ndarray)):
            if x2 is None:
                x2 = x1.copy()
            else:
                # What is it? - raise exception
                raise ValueError

        elif isinstance(x2, (float, list)):
            x2 = np.asarray(x2)

        T, S = np.meshgrid(x2, x1)
        return self.kfunc(ind1, ind2,
                          S.ravel(), T.ravel(),
                          kpar).reshape(T.shape)


##
# Special instance of the Multioutput framework
# where the outputs correspond to Y(x1) and dY/dx(x2)
class GradientMultioutputKernel(MultioutputKernel):
    def __init__(self, kfunc, kpar=None):
        super(GradientMultioutputKernel, self).__init__(kfunc, kpar)

    """
    Example useage for the square exponential kernel

    >>> ksqexp = GradientMultioutputKernel.SquareExponKernel()
    >>> s = [0., 0.5]
    >>> t = [0., 0.5, 1.]
    >>> ksqexp.cov(0, 1, s, t)
    # returns
    #  np.array([[ cov{ X(x1) dX/dt(x2) for x2 in t ] for x1 in s]])
    #
    """

    @classmethod
    def SquareExponKernel(cls, kpar=None):
        if not isinstance(kpar, np.ndarray):
            if kpar is None:
                kpar = (1., 1.)

        def kxx(s, t, par):
            return par[0]*np.exp(-par[1]*(s-t)**2)

        def kxdx(s, t, par):
            return (2*par[0]*par[1]*(s-t)*np.exp(-par[1]*(s-t)**2))

        def kdxdx(s, t, par):
            return (2*par[0]*par[1]*(1-2*par[1]*(s-t)**2)*np.exp(
                        -par[1]*(s-t)**2))

        def k(ind1, ind2, t1, t2, par):
            if ind1 == 0 and ind2 == 0:
                return kxx(t1, t2, par)
            elif ind1 == 0 and ind2 == 1:
                return kxdx(t1, t2, par)
            elif ind1 == 1 and ind2 == 0:
                return kxdx(t2, t1, par)
            elif ind1 == 1 and ind2 == 1:
                return kdxdx(t1, t2, par)

        return cls(k, kpar)
