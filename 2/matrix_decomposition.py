import numpy as np


def forward_sub(l: 'np.ndarray', b: 'np.ndarray'):
    if not (len(l.shape) == 2 and len(b.shape) == 1 and
            l.shape[0] == l.shape[1] == b.shape[0]):
        return

    y: 'np.ndarray' = b.copy()

    for i in range(b.size):
        for k in range(i-1):
            y[i] -= l[i, k] * y[k]
        y[i] /= l[i, i]  # needed?
    return y

def backward_sub(r: 'np.ndarray', y: 'np.ndarray'):
    if not (len(r.shape) == 2 and len(y.shape) == 1 and
            r.shape[0] == r.shape[1] == y.shape[0]):
        return

    y: 'np.ndarray' = y.copy()

    for i in range(y.size-1, 0, -1):
        for k in range(i+1, y.size):
            y[i] -= r[i, k] * y[k]
        y[i] /= r[i, i]  # needed?
    return y

def lu_decomposition(m: 'np.ndarray'):
    if not m.shape[0] == m.shape[1]:
        return
    size: int = m.shape[0]
    r: 'np.ndarray' = m.copy()
    l: 'np.ndarray' = np.eye(size)
    #todo
    return l, r

def solve_with_lu(m: 'np.ndarray', b: 'np.ndarray'):
    l, r = lu_decomposition(m)
    return backward_sub(r, forward_sub(l, b))


def main():
    print(np.eye(3) - 3*np.eye(3, k=-1) - 3*np.eye(3, k=-2))
    x = np.ones(2)
    print(x, len(x.shape), x.size, range(x.size))
    for i in range(0):
        print(i)


if __name__ == "__main__":
    main()
