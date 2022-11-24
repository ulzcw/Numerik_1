import numpy as np


def check_matrix_shape(M: 'np.ndarray') -> None:
    if len(M.shape) != 2 or M.shape[0] != M.shape[1]:
        raise Exception("Matrix needs to be square")


def check_matrix_vector_shape(M: 'np.ndarray', v: 'np.ndarray') -> None:
    check_matrix_shape(M)

    if len(v.shape) != 1:
        raise Exception(f"{v} is not a vector")

    if M.shape[0] != v.shape[0]:
        raise Exception("Matrix and vector are not of the right shape")


def forward_sub(L: 'np.ndarray', b: 'np.ndarray'):
    check_matrix_vector_shape(L, b)
    if L.dtype != 'float64':
        L = L.astype('float64')

    y: 'np.ndarray' = b.astype('float64')

    for i in range(y.size):
        for k in range(i):
            y[i] -= L[i, k] * y[k]
        y[i] /= L[i, i]  # needed?
    return y


def backward_sub(R: 'np.ndarray', y: 'np.ndarray'):
    check_matrix_vector_shape(R, y)
    if R.dtype != 'float64':
        R = R.astype('float64')

    x: 'np.ndarray' = y.astype('float64')

    for i in range(x.size - 1, -1, -1):  # last to first x[ ]
        for k in range(i + 1, x.size):
            x[i] -= R[i, k] * x[k]
        x[i] /= R[i, i]
    return x


def lu_decomposition(M: 'np.ndarray'):
    check_matrix_shape(M)

    size: int = M.shape[0]
    r: 'np.ndarray' = M.copy()
    l: 'np.ndarray' = np.eye(size)
    for step in range(size - 1):
        for row in range(step + 1, size):
            l[row, step] = r[row, step] / r[step, step]
            r[row, :] = r[row, :] - l[row, step] * r[step, :]

    return l, r


def solve_with_lu(M: 'np.ndarray', b: 'np.ndarray'):
    check_matrix_shape(M)

    L, R = lu_decomposition(M.astype('float64'))
    return backward_sub(R, forward_sub(L, b.astype('float64')))


def main():
    print(np.eye(3) - 3 * np.eye(3, k=-1) - 3 * np.eye(3, k=-2))
    x = np.ones(2)
    print(x, len(x.shape), x.size, range(x.size))
    for i in range(0):
        print(i)

    L = np.eye(3) - 2 * np.eye(3, k=-1) - 2 * np.eye(3, k=-2)
    R = np.array([[2, -1, -2],
                  [0, 4, -1],
                  [0, 0, 8]])
    M = L.dot(R)
    b = np.array([1, 1, 1])
    x, y = lu_decomposition(M)
    print("", M, x, y, sep="\n\n")
    print(solve_with_lu(M, b))


if __name__ == "__main__":
    main()
