import numpy as np

class Kernel:
    def __init__(self, kfunc, kpar=None):
        self.kfunc = kfunc
        self.kpar = kpar


    def __call__(self, t1, t2):
        return self.kfunc(t1, t2, self.kpar)
    

    def cov(self, t1, t2=None, kpar=None):
        if kpar is None:
            kpar = self.kpar
            
        if isinstance(t1, np.ndarray):
            if t2 is None:
                t2 = t1.copy()
            T1, T2 = np.meshgrid(t1, t2)
            return self.kfunc(T1.ravel(), T2.ravel(), kpar).reshape(T1.shape)

###
# ToDo 
# extend the arguments to include
#
#    "kpar_shared" - representing those parameters shared by all kernels
#    "kpari"       - representing those parameters distinct to kernel i
#
# Inputs are currently usually written as t1 etc. indicative of the fact
# most of this was written with inference in dynamic systems in mind, it
# should still be very general for multivariate inputs after appropriate
# ravelling, which is something that would need to be taken care of the
# kfunc
class CollectionKernel:
    def __init__(self, kfunc, size, kpar=None):
        self.kfunc = kfunc
        self.kpar = kpar
        self.size = size

    def __call__(self, t1, t2, ind1, ind2, **kwargs):
        if not "kpar" in kwargs.keys():
            kwargs["kpar"] = self.kpar

        return self.kfunc(t1, t2, ind1, ind2, **kwargs)

    def cov(self, t1, t2=None, ind1=None, ind2=None, **kwargs):
        ###
        # If ind1 == ind2 == None returns the complete covariance matrix

        # else returns the covariance between ind1 and ind2
        if isinstance(ind1, int) and isinstance(ind2, int):
            if not isinstance(t2, np.ndarray):
                t2 = t1.copy()

            T_, S_ = np.meshgrid(t1, t2)

            return self.__call__(S_.ravel(), T_.ravel(), ind1, ind2, **kwargs).reshape(T_.shape)
        # Not yet implemented:
        # - Allow the complete covariance function
        #
        #  | C[0,0] C[0,1] C[1,2] ... |  
        #  |    -   C[1,1] C[1,2] ... | C[i, j] = self.cov(t1, t2, i, j, ...)
        #  |    -      -   C[2,2] ... |  
        #  |                  .    .  |
        #
        # but it may be the case that each of the outputs {y_i}  in the collection has a
        # different collection of inputs {t_{i}} associated with it, this is probably better
        # left to the class that makes use of the kernel

        
def SquareExponKernel():                
    kfunc = lambda s, t, par: par[0]*np.exp(-par[1]*(s-t)**2)
    cls = Kernel(kfunc, [1., 1.])
    return cls
