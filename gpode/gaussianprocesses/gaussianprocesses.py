import gpode.kernels.kernels as kernels


class GaussianProcess:
    def __init__(self, kernel, inputs=None):

        if isinstance(kernel, kernels.Kernel):
            self.kernel = kernel
        else:
            raise ValueError

        self.inputs = inputs

        # store Cholesky decomp. of covar mat during fitting
        self.L = None


class MultioutputGaussianProcess(GaussianProcess):
    def __init__(self, mokernel, inputs=None):

        if isinstance(mokernel, kernels.MultioutputKernel):
            super(MultioutputGaussianProcess, self).__init__(mokernel, inputs)

            self.Ls = {}
            self.cross_covars = {}
            self.fitted_outputs = {}

        # Initalisation failed
        else:
            raise ValueError

    # Will usually be overridden in subclasses
    def fit(self, outputs):
        self._fit(outputs)

    ##
    # Puts the model in a position to return predictive
    # distributions as desired
    def _fit(self, outputs):
        _outputs_w_data = []  # Which outputs have been fitted
        for ind, data in outputs:
            _outputs_w_data.append(ind)
            self.fitted_outputs[ind] = data

        _outputs_w_data = sorted(_outputs_w_data)
        for i in _outputs_w_data:
            for j in _outputs_w_data:
                pass
                #self.fit_cov(i, j)
            _outputs_w_data.remove(i)


def gradgp_fit(func):
    def func_wrapper(self, *args, **kwargs):
        data = []
        for ind, key in zip((0, 1), ("x", "dx")):
            try:
                data.append((ind, kwargs[key]))
            except:
                pass
        self._fit(data)
    return func_wrapper


class GradientGaussianProcess(MultioutputGaussianProcess):
    def __init__(self, kernel, inputs=None):
        super(GradientGaussianProcess, self).__init__(kernel, inputs)

    @gradgp_fit
    def fit(self, x=None, dx=None):
        pass


import numpy as np
k = kernels.GradientMultioutputKernel.SquareExponKernel()
ss = np.linspace(0., 1., 3)

gp = GradientGaussianProcess(k, inputs=ss)
gp.fit(x=[0, 1, 2])
