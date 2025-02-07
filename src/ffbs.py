
import numpy as np
import numba as nb


# Define FFBS
@nb.njit(cache=True)
def FFBS(lhood, transition_matrix):

    # Get parameters
    num_states, num_data = lhood.shape
    pi0 = np.ascontiguousarray(transition_matrix[-1, :])
    pis = np.ascontiguousarray(transition_matrix[:-1, :])
    states = np.zeros(num_data, dtype=np.int32)

    # Forward filter
    forward = np.zeros((num_data, num_states)).T
    forward[:, 0] = lhood[:, 0] * pi0
    forward[:, 0] /= np.sum(forward[:, 0])
    for n in range(1, num_data):
        forward[:, n] = lhood[:, n] * (pis.T @ forward[:, n - 1])
        forward[:, n] /= np.sum(forward[:, n])

    # Backward sample
    s = np.searchsorted(np.cumsum(forward[:, -1]), np.random.rand())
    states[-1] = s
    for m in range(1, num_data):
        n = num_data - m - 1
        backward = forward[:, n] * pis[:, s]
        backward /= np.sum(backward)
        s = np.searchsorted(np.cumsum(backward), np.random.rand())
        states[n] = s

    return states



