import numpy as np
import scipy.stats
from collections import OrderedDict


class ProbabilityDistribution:
    def __init__(self, pdf=None, rvs=None):
        self.pdf = pdf
        self.rvs = rvs


class ProposalDistribution(ProbabilityDistribution):
    def __init__(self,
                 pdf=None, rvs=None,
                 is_symmetric=False):
        super(ProposalDistribution, self).__init__(pdf, rvs)
        self.is_symmetric = is_symmetric


##
# These are all a little weird at the moment in that we are
# attaching the parameters of the prior and proposal the
# parameter cls object
def handle_prior_assignment(cls, tup):
    if tup[0] == "gamma":
        cls.prior_hyperpar = tup[1]
        cls.prior = scipy.stats.gamma(a=cls.prior_hyperpar[0],
                                      scale=cls.prior_hyperpar[1])
    elif tup[0] == "unif":
        cls.prior_hyperpar = tup[1]
        cls.prior = scipy.stats.uniform(loc=tup[1][0],
                                        scale=tup[1][1]-tup[1][0])
    else:
        raise ValueError


def handle_proposal_assignment(cls, tup):
    if tup[0] == "normal rw":
        cls.proposal_hyperpar = tup[1]
        q = ProposalDistribution(
            rvs=lambda xcur: np.random.normal(loc=xcur,
                                              scale=cls.proposal_hyperpar),
            is_symmetric=True)

        cls.proposal = q


###
# Parameter class definitions etc
class Parameter:
    def __init__(self, name, prior=None, proposal=None, value=None):
        self.name = name

        if isinstance(prior, tuple):
            handle_prior_assignment(self, prior)
        else:
            self.prior = prior

        if isinstance(proposal, tuple):
            handle_proposal_assignment(self, proposal)
        else:
            self.proposal = proposal

        self.value = value


class ParameterCollection:
    def __init__(self, parameters):
        self.parameters = OrderedDict((p.name, p) for p in parameters)

    def value(self, np_arrayfy=False, arr_shape=None):
        v = [item[1].value for item in self.parameters.items()]

        if np_arrayfy:
            v = np.array(v)
            if arr_shape is not None:
                v.shape = arr_shape

        return v
