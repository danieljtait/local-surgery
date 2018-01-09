import numpy as np

#################################################
# If x1, x2 are jointly normal distributed then
# returns the parameters of the distribution
#
#     p(x1 | x2) ~ N(condMean, condCov)
#
def CondPar(x2, m1, m2, C11, C12, C22):
    L = np.linalg.cholesky(C22)
    condMean = m1 + np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, x2-m2)))
    condCov = C11 - np.dot(C12, np.linalg.solve(L.T, np.linalg.solve(L, C12.T)))
    return condMean, condCov


####
# Simulates a single sample from the multivariate
# normal distribution
def Sim(mean, cov=None, L=None):
    # Looking for the Cholesky decomposition to have been passed
    # as an argument
    if L:
        return mean + np.dot(L, np.random.normal(size=L.shape[0]))
    else:
        L = np.linalg.cholesky(cov)
        return Sim(mean, L=L)
