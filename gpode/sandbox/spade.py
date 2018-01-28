from gpode.bayes import Parameter, ParameterCollection
from gpode.kernels import Kernel


p1 = Parameter("p1", prior=("gamma", (4, 0.2)))
p2 = Parameter("p2", prior=("gamma", (4, 0.2)))


phi_k = ParameterCollection([p1, p2])
for pname, par in phi_k.parameters.items():
    par.value = par.prior.rvs()


kse = Kernel.SquareExponKernel(kpar=phi_k)
print(kse.cov(0.4, 0.3).shape)
