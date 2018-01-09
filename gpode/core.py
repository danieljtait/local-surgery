from . import multivariatenormal


class GaussianProcess:
    def __init__(self, kernel):
        self.kernel = kernel

        # Store any fitted values
        self._fittedValues = {'t': None,
                              'x': None,
                              'C': None,
                              'L': None }


    def fit(self, inputPoints, dataPoints, storeCovL=True):
        self._fittedValues['t'] = inputPoints
        self._fittedValues['x'] = dataPoints

        C = self.kernel.Cov(inputPoints)
        if storeCovL:
            L = np.linalg.cholesky(C)
            self._fittedValues['L'] = L


    def pred(self, newinputs):
        L = self._fittedValues['L']
        Cinva = np.linalg.solve(L.T, np.linalg.solve(L, self._fittedValues['x']))
        k = self.kernel(newinputs, self._fittedValues['t'])
        return np.dot(k, Cinva)
