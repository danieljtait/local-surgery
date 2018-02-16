import numpy as np

#####
#
# if X ~ N(mean, cov) and X = Az + b then we
# rearrange for the mean and covariance of z
#
# (possibly degenerate)
def mvt_linear_trans(A, b, mean, cov=None, inv_cov=None, return_type="inv"):
    if inv_cov is None:
        inv_cov = np.linalg.inv(cov)
        
    z_inv_cov = np.dot(A.T, np.dot(inv_cov, A))
    z_mean = np.linalg.solve(A, mean - b)

    if return_type == "inv":
        return z_mean, z_inv_cov
    else:
        return z_mean, np.linalg.inv(z_inv_cov)
