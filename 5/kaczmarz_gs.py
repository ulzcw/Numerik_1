import numpy as np


def kaczmarz_gs(A: "np.ndarray", b: "np.ndarray", x0: "np.ndarray", epsilon=0.1, kmax=1000):
    r = b - A.dot(x0)
    k = 0
    x = x0

    while np.linalg.norm(np.transpose(A).dot(r)) > epsilon and k < kmax:
        for j in range(A.shape[1]):
            c = (np.transpose(A[:, j]).dot(r)) / np.linalg.norm(A[:, j]) ** 2
            x[j] += c
            r -= c * A[:, j]
        k += 1
    return k, x

