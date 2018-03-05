import numpy as np

### Handling of the covariance function


def get_cov(i, ta, tb, cov_funcs):

    assert(ta.size == tb.size)

    C00 = _get_cov_pq(0, 0, ta, tb, cov_funcs, 1, 1)
    C01 = _get_cov_pq(0, 1, ta, tb, cov_funcs, 1, 1)
    C11 = _get_cov_pq(1, 1, ta, tb, cov_funcs, 1, 1)

    N = ta.size

    C = np.arange(0, 25).reshape(5, 5)
    C00 = C[:2, :2]
    C01 = C[:2, 2:]
    C22 = C[:2, :2]
    
    n_ind = [j for j in range(N) if j != i]

    C00_ii = C00[i, i]
    C11_ii = C11[i, i]

    C01_ij = C01[i, n_ind]
    C11_ij = C11[i, n_ind]

    C22aa = C11[n_ind, :]
    C22aa = C22aa[:, n_ind]
    C22ab = C01[n_ind, :]
    C22ab = C22ab[:, n_ind]
    C22bb = C11[n_ind, :]
    C22bb = C11[:, n_ind]


def _get_cov_pq(p, q, vec_ta, vec_tb, cov_funcs, theta0, theta1):

    _Ta, _Sa = np.meshgrid(vec_ta, vec_ta)
    _Tb, _Sb = np.meshgrid(vec_tb, vec_tb)

    key = "J{}J{}".format(p, q)

    return cov_funcs[key](_Sb.ravel(), _Tb.ravel(),
                          _Sa.ravel(), _Ta.ravel(),
                          theta0, theta1).reshape(_Ta.shape)

def _cond_partition(N, i, A, B, C):
    M11 = np.array([[A[i, i], B[i, i]],
                    [B[i, i], C[i, i]]])

    n_ind = np.array([j for j in range(N) if j != i])

    M12 = np.row_stack((np.concatenate((A[i, n_ind], B[i, n_ind])),
                        np.concatenate((B[i, n_ind].T, C[i, n_ind]))))

    M22 = np.row_stack((np.column_stack((A[n_ind[:, None], n_ind], B[n_ind[:, None], n_ind])),
                        np.column_stack((B[n_ind[:, None], n_ind].T, C[n_ind[:, None], n_ind]))))

    return M11, M12, M22

N = 3
i = 1

M = np.arange(0, 36).reshape(6, 6)
A = M[:3, :3]
B = M[:3, 3:]
C = M[3:, 3:]

M11, M12, M22 = _cond_partition(N, i, A, B, C)
print(M)
print(M11)
print(M12)
print(M22)

    
    
