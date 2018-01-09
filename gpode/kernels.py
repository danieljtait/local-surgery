import numpy as np

class Kernel:
    def __init__(self, kfunc, kpar=None):
        self.kfunc = kfunc
        self.kpar = kpar


    def __call__(self, t1, t2):
        return self.kfunc(t1, t2, self.kpar)
    

    def Cov(self, t1, t2=None):
        if isinstance(t1, np.ndarray):
            if t2 is None:
                t2 = t1.copy()
            T1, T2 = np.meshgrid(t1, t2)
            return self.kfunc(T1.ravel(), T2.ravel(), self.kpar).reshape(T1.shape)


class CollectionKernel:
    def __init__(self, kfunc, size, kpar=None):
        self.kfunc = kfunc
        self.kpar = kpar
        self.size = size

    def __call__(self, t1, t2, ind1, ind2):
        return self.kfunc(t1, t2, ind1, ind2, **self.kpar)


def SquareExponKernel():                
    kfunc = lambda s, t, par: par[0]*np.exp(-par[1]*(s-t)**2)
    cls = Kernel(kfunc, [1., 1.])
    return cls
