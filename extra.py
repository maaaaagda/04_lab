from numpy import array, dot, mean, std, empty, argsort, dtype, sum
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show

def cov(X):
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """
    return dot(X.T, X) / X.shape[0]
    # N = data.shape[1]
    # C = empty((N, N))
    # for j in range(N):
    #   C[j, j] = mean(data[:, j] * data[:, j])
    #   for k in range(j + 1, N):
    #       C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    # return C

def pca(data, pc_count = None):
    """
    Principal component analysis using eigenvalues
    note: this mean-centers and auto-scales the data (in-place)
    """
    data = array(data, dtype="float64")
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(data, V)  # used to be dot(V.T, data.T).T
    return U, E, V

