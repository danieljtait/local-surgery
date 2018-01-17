from . import multivariatenormal

import numpy as np

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
        if isinstance(newinputs, float):
            newinputs = [newinputs]
            
        L = self._fittedValues['L']
        Cinva = np.linalg.solve(L.T, np.linalg.solve(L, self._fittedValues['x']))
        k = np.array([self.kernel(t, self._fittedValues['t']) for t in newinputs])
        return np.dot(k, Cinva)


####
# Class for a collection of Gaussian processes
# [y_1(t),...,y_Q(t) ]
#
# the collectionKernel.__call__() method must have
# arguments ind1, ind2 representing the cross covariance
# between output y_ind1 and y_ind2
class GaussianProcessCollection:
    def __init__(self, collectionKernel):
        self.kernel = collectionKernel

        self.fittedValues = {'ts': None,
                             'xs': None,
                             'C': None,
                             'L': None }

    def fit(self, inputPoints, dataPoints):
        if isinstance(inputPoints, list):
            if len(inputPoints, list):
                self.fittedValues['ts'] = inputPoints

        else:
            self.fittedValues['ts'] = [inputPoints for i in range(self.kernel.size)]

        self.fittedValues['xs'] = dataPoints


    def cov(self):
        # Override .kernel(...) for 
        def _cov(s, t, p, q):
            T, S = np.meshgrid(t, s)
            if p==q:
                return 0.5*self.kernel(S.ravel(), T.ravel(), ind1=p, ind2=q).reshape(S.shape)
            else:
                return self.kernel(S.ravel(), T.ravel(), ind1=p, ind2=q).reshape(S.shape)

        ts = self.fittedValues['ts']
        N = sum(t.size for t in ts)
        result = np.zeros((N, N))
        nc = 0
        nr = 0
        for p, t1 in enumerate(ts):
            r = np.column_stack((_cov(t1, t2, p, q)
                                 for q, t2 in enumerate(ts[:p+1])))
            result[nc:nc+t1.size, :nr+t1.size] = r
            nc += t1.size
            nr += t1.size
        C = result + result.T
        print(C==C.T)
        print(np.linalg.eig(C)[0])
        self.fittedValues['L'] = np.linalg.cholesky(C)
        return C
