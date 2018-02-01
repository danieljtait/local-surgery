import numpy as np
from .nestedgpintegrals import ngpintegrals_sqexpcovs
from gpode.kernels import MultioutputKernel


class NestedIntegralKernel(MultioutputKernel):

    def __new__(cls, kpar=[1., 1.], t0=0., origin="fixed"):
        if origin == "recursive":
            return NestedIntegralKernelRecursive(kpar)
        elif origin == "fixed":
            return NestedIntegralKernelFixed(kpar)
        else:
            raise ValueError


class NestedIntegralKernelFixed(MultioutputKernel):

    def __init__(self, kpar=[1., 1.], t0=0.):

        self.origin = "fixed"

        _se_int_covs = ngpintegrals_sqexpcovs()
        self._se_int_covs = ngpintegrals_sqexpcovs()

        self._t0 = t0

        def _kfunc(i, j, s, t, kpar, **kwargs):

            if i <= j:
                key = "J{}J{}".format(i, j)
                return _se_int_covs[key](s, t,
                                         self._t0, self._t0,
                                         kpar[0], kpar[1])
            else:
                return _kfunc(j, i, t, s, kpar, **kwargs)

            super(NestedIntegralKernelFixed, self).__init__(_kfunc, kpar)


class NestedIntegralKernelRecursive(MultioutputKernel):
    def __init__(self, kpar):
        self.origin = "recursive"
        self.kpar = kpar

        _se_int_covs = ngpintegrals_sqexpcovs()

        def _kfunc(i, j, sb, tb, sa, ta, kpar):
            if i > j:
                return _kfunc(j, i, tb, sb, ta, sa, kpar)
            else:
                key = "J{}J{}".format(i, j)
                return _se_int_covs[key](sb, tb,
                                         sa, ta,
                                         kpar[0], kpar[1])

        super(NestedIntegralKernelRecursive, self).__init__(_kfunc, kpar)

    # cov needs to be overridden compared to the usual MultioutputKernel
    # to handle the initial conditions
    def cov(self, ind1, ind2, sb, sa, tb=None, ta=None, kpar=None):

        kpar = self._parse_kernel_par(kpar)

        if not isinstance(sb, np.ndarray):
            sb = np.asarray(sb)

        if not isinstance(sa, np.ndarray):
            sa = np.asarray(sa)

        if not isinstance(tb, (float, list, np.ndarray)):
            if tb is None:
                tb = sb.copy()
            else:
                raise ValueError

        elif isinstance(ta, (float, list)):
            tb = np.asarray(tb)

        if not isinstance(ta, (float, list, np.ndarray)):
            if ta is None:
                ta = sa.copy()
            else:
                raise ValueError

        elif isinstance(ta, (float, list)):
            ta = np.asarray(ta)

        Tb, Sb = np.meshgrid(tb, sb)
        Ta, Sa = np.meshgrid(ta, sa)

        return self.kfunc(ind1, ind2,
                          Sb.ravel(), Tb.ravel(),
                          Sa.ravel(), Ta.ravel(),
                          kpar).reshape(Sb.shape)
