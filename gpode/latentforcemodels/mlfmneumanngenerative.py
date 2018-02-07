import numpy as np
from .nestedgpintegrals import ngpintegrals_sqexpcovs
from gpode.kernels import MultioutputKernel


class Data:
    def __init__(self, time, Y):
        self.time = time
        self.Y = Y


class NeumannGenerativeModel:

    def __init__(self, As,
                 data_time, data_Y,
                 origin_type="fixed"):
        self.origin_type = origin_type
        self.data = Data(data_time, data_Y)

    def _forward_integrate(self, x0, tt, J1J2):
        if self.origin_type == "fixed":
            pass
        elif self.origin_type == "recursive":
            pass

    def _set_times(self, t0_ind=None):
        if t0_ind is None:
            t0_ind = 0

        self._t0_ind = t0_ind

        tt = self.data.time
        Y = self.data.Y

        self._t0 = tt[self._t0_ind]

        forward_times = tt[tt >= self._t0]
        backward_times = tt[tt < self._t0]
        print(backward_times)
        print(forward_times)


def _integrate_recursive_forward(tb_vec, ta_vec, J1J2, x0, A, B):
    # Make sure the sequence is sorted properly
    #  - should all potentially check that
    #    tb[i] <= tb[i+1] && ta[i] <= ta[i+1]
    #    but perhaps just trust the input is proper
    # assert(all(tb_vec >= ta_vec))
    AA = np.dot(A, A)
    AB = np.dot(A, B)
    BA = np.dot(B, A)
    BB = np.dot(B, B)

    X = [x0]
    for nt, (tb, ta) in enumerate(zip(tb_vec, ta_vec)):

        j1 = J1J2[nt, 0]
        j2 = J1J2[nt, 1]

        x1new = np.dot(A*(tb-ta) + B*J1J2[nt, 0], X[-1])
        x2new = np.dot(AB*j2 + BA*((tb-ta)*j1 - j2), X[-1])
        x3new = 0.5*np.dot(AA*(tb-ta)**2 + BB*J1J2[nt, 0]**2, X[-1])

        X.append(X[-1] + x1new + x2new + x3new)

    return np.array(X)


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
