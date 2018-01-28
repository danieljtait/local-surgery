import numpy as np
from gpode.bayes import Parameter, ParameterCollection
from gpode.kernels import Kernel


p1 = Parameter("p1",
               prior=("gamma", (4, 0.2)),
               proposal=("normal rw", (0.1,)))

p2 = Parameter("p2", prior=("gamma", (4, 0.2)),
               proposal=("normal rw", (0.1,)))


phi_k = ParameterCollection([p1, p2], independent=True)
for pname, par in phi_k.parameters.items():
    par.value = par.prior.rvs()

kse = Kernel.SquareExponKernel(kpar=phi_k)
print(phi_k.value())
print(np.array(phi_k.proposal.rvs(phi_k.value())))
