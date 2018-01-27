import gpode.kernels.kernels as kernels
import numpy as np


class GaussianProcess:
    def __init__(self, kernel, inputs=None):

        if isinstance(kernel, kernels.Kernel):
            self.kernel = kernel
        else:
            raise ValueError

        self.inputs = inputs

        # store Cholesky decomp. of covar mat during fitting
        self.L = None

    def fit(self, X, y):
        C = self.kernel.cov(X)
        self.L = np.linalg.cholesky(C)
        self.training_data = (X, y)
        self._Cinv_a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))

    def pred(self, X):
        C12 = self.kernel.cov(X, self.training_data[0])
        return np.dot(C12, self._Cinv_a)


class MultioutputGaussianProcess(GaussianProcess):
    def __init__(self, mokernel, inputs=None):

        if isinstance(mokernel, kernels.MultioutputKernel):
            super(MultioutputGaussianProcess, self).__init__(mokernel, inputs)

            self.Ls = {}
            self.cross_covars = {}
            self.fitted_outputs = {}

            self._ind_map = lambda i: i

        # Initalisation failed
        else:
            raise ValueError

    # Will usually be overridden in subclasses
    def fit(self, outputs):
        self._fit(outputs)

    def pred(self, inds, inputs=None):
        self._pred(inds, inputs)

    def get_full_cov(self, inputs):
        result = None
        inds = [i for i in inputs.keys()]
        while inds != []:
            ind = inds[0]
            row = [0.5*self.kernel.cov(ind, ind,
                                       inputs[ind], inputs[ind])]
            inds.remove(ind)
            for ind2 in inds:
                C12 = self.kernel.cov(ind, ind2,
                                      inputs[ind], inputs[ind2])
                row.append(C12)
            row = np.column_stack((M for M in row))
            if isinstance(result, np.ndarray):
                Mzero = np.zeros((row.shape[0], result.shape[1]-row.shape[1]))
                row = np.column_stack((Mzero, row))
                result = np.row_stack((result, row))
            else:
                result = row

        result = result + result.T

        return result

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
                self._fit_cov(i, j)
            _outputs_w_data.remove(i)

    def _fit_cov(self, ind1, ind2):
        if ind1 == ind2:
            C11 = (self.kernel.cov(self._ind_map(ind1),
                                   self._ind_map(ind1),
                                   self.inputs[ind1]))
            self.Ls[ind1] = np.linalg.cholesky(C11)
        else:
            C12 = self.kernel.cov(self._ind_map(ind1),
                                  self._ind_map(ind2),
                                  self.inputs[ind1],
                                  self.inputs[ind2])

            self.cross_covars[(ind1, ind2)] = C12

    def _pred(self, inds, inputs):
        # Either predict ind i at inputs

        # or else predict at new input
        pass


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
    def __init__(self, kernel, inputs=None, same_inputs=False):
        if same_inputs:
            inputs = {"x": inputs, "dx": inputs}
        super(GradientGaussianProcess, self).__init__(kernel, inputs)

        def _imap(i):
            if isinstance(i, int):
                return i
            elif i == "y":
                return 0
            elif i == "dy":
                return 1

        self._ind_map = _imap

#    @gradgp_fit
#    def fit(self, x=None, dx=None):
#        pass

    def fit(self,
            y_data=None,
            dy_data=None):
        try:
            if isinstance(y_data[1], float):
                y_data[1] = np.array([y_data[1]])
        except:
            pass

        try:
            if isinstance(dy_data[1], float):
                dy_data[1] = np.array([dy_data[1]])
        except:
            pass
                
        self.training_data = {"y": y_data,
                              "dy": dy_data}

        for key, item in self.training_data.items():
            if item is not None:
                self.fit_cov(key, key)

        self._trained_inds = [key for key, item in self.training_data.items()
                              if item is not None]

        _trained_inds_cp = [val for val in self._trained_inds]
        for key in _trained_inds_cp:
            self.fit_cov(key, key)
            _trained_inds_cp.remove(key)
            for key2 in _trained_inds_cp:
                self.fit_cov(key, key2)

    def fit_cov(self, ind1, ind2):
        if ind1 == ind2:
            ind_alias = self._ind_map(ind1)
            C = self.kernel.cov(ind_alias,
                                ind_alias,
                                self.training_data[ind1][0])
            self.Ls[ind1] = np.linalg.cholesky(C)
        else:
            ind1_alias = self._ind_map(ind1)
            ind2_alias = self._ind_map(ind2)
            C = self.kernel.cov(ind1_alias,
                                ind2_alias,
                                self.training_data[ind1][0],
                                self.training_data[ind2][0])
            self.cross_covars[(ind1, ind2)] = C

    ###
    # Constructs the full covariance function
    # from all the trained data - should really be overridden
    # in this class to take advantage of their only being two
    # outputs
    def get_full_cov(self):
        inds = [val for val in self._trained_inds]
        result = None
        while inds != []:
            ind = inds[0]
            row = [0.5*np.dot(self.Ls[ind], self.Ls[ind].T)]
            inds.remove(ind)
            for ind2 in inds:
                Cij = self.cross_covars[(ind, ind2)]
                row.append(Cij)
            row = np.column_stack((M for M in row))
            if isinstance(result, np.ndarray):
                Mzero = np.zeros((row.shape[0], result.shape[1]-row.shape[1]))
                row = np.column_stack((Mzero, row))
                result = np.row_stack((result, row))
            else:
                result = row

        result = result + result.T
        return result

    def predict(self, y_inputs=None, dy_inputs=None, ret_par=False):
        if y_inputs is not None and dy_inputs is None:
#            C11 = self.kernel.cov(0, 0, y_inputs)

            C12 = [self.kernel.cov(0, self._ind_map(ind),
                                   y_inputs, self.training_data[ind][0])
                   for ind in self._trained_inds]

            C12 = np.column_stack((_C for _C in C12))

            L22 = np.linalg.cholesky(self.get_full_cov())
            a = [self.training_data[ind][1] for ind in self._trained_inds]
            a = np.concatenate([_a for _a in a])

            Cinva = np.linalg.solve(L22.T, np.linalg.solve(L22, a))
            ypred = np.dot(C12, Cinva)

            if ret_par:
                C11 = self.kernel.cov(0, 0, y_inputs)
                C11cond = C11 - np.dot(C12,
                                       np.linalg.solve(
                                           L22, np.linalg.solve(L22, C12.T)))
                return ypred, C11cond
            else:
                return ypred

        elif y_inputs is None and dy_inputs is not None:
            C12 = [self.kernel.cov(self._ind_map(ind), 1,
                                   self.training_data[ind][0],
                                   dy_inputs)
                   for ind in self._trained_inds]
            C12 = np.column_stack((_C for _C in C12))
            L11 = np.linalg.cholesky(self.get_full_cov())

            a = [self.training_data[ind][1] for ind in self._trained_inds]
            a = np.concatenate([a_ for a_ in a])

            Cinva = np.linalg.solve(L11.T, np.linalg.solve(L11, a))
            ypred = np.dot(C12.T, Cinva)

            if ret_par:
                C22 = self.kernel.cov(1, 1, dy_inputs)
                C22cond = C22 - np.dot(C12.T,
                                       np.linalg.solve(
                                           L11.T, np.linalg.solve(L11, C12)))
                return ypred, C22cond
            else:
                return ypred
