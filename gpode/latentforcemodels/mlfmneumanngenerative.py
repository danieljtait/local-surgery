from .nestedgpintegrals import ngpintegrals_sqexpcovs
from gpode.kernels import MultioutputKernel


class NestedIntegralKernel(MultioutputKernel):
    def __init__(self, kpar=[1., 1.], t0=0., origin="fixed"):

        _se_int_covs = ngpintegrals_sqexpcovs()

        if origin == "fixed":

            self._t0 = t0
            self._origin = "fixed"

            def _kfunc(i, j, s, t, kpar):
                if i <= j:
                    key = "J{}J{}".format(i, j)
                    return _se_int_covs[key](s, t,
                                             self._t0, self._t0,
                                             kpar[0], kpar[1])

                else:
                    return _kfunc(j, i, t, s, kpar)

            super(NestedIntegralKernel, self).__init__(_kfunc, kpar)

        else:
            raise NotImplementedError
