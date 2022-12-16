import time
from typing import Iterable, Callable

import numpy as np
from numpy.linalg import norm
from scipy.linalg import null_space


# L_nk: "np.ndarray" = np.zeros((12, 12))

L: "np.ndarray" = np.asarray([
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    ])

N = L.shape[0]


def make_a(a: float):

    _L: "np.ndarray" = L.copy()
    D: "np.ndarray" = np.zeros(N)

    for i in range(N):  # evt todo optimise
        # for k in range(N):
        D[i] = sum(_L[i])
        if not D[i]:
            # for k in range(N):
            #     _L[i, k] = 1
            _L[i] = np.ones(N)
            D[i] = N
        D[i] = 1 / D[i]

    D = np.diag(D)

    return (1 - a) * _L.transpose().dot(D) + (a / N) * np.ones(_L.shape)


def potenzmethode_alt(start_v: "np.ndarray", A: "np.ndarray", *, n: int = 0):
    _v = start_v

    if n > 1:
        _v = potenzmethode_alt(_v, A, n=n-1)

    Av: "np.ndarray" = A.dot(_v)
    print(norm(Av))
    return Av / norm(Av)


def potenzmethode(A: "np.ndarray",  start_v: "np.ndarray" = np.ones(N) / N, *, eps: float = 10**(-6)):
    Av: "np.ndarray"
    _v: "np.ndarray" = start_v
    Av_norm = 1 + 10 * eps

    while abs(Av_norm - 1) > eps:
        Av = A.dot(_v)
        Av_norm = norm(Av)
        _v = Av / Av_norm
        # print(Av_norm)

    # print(Av_norm)
    # print(_v)
    return _v


# def null_spac


def speed_test(func: Callable, args: Iterable):
    _time = time.perf_counter_ns()
    for _arg in args:
        # print(func(_arg))
        func(_arg)
    return time.perf_counter_ns() - _time


if __name__ == "__main__":
    I = np.eye(N)

    a_1 = 0.1
    a_2 = 0.3
    a_3 = 0.6

    a_set = (make_a(a_1), make_a(a_2), make_a(a_3))
    b_set = (make_a(a_1)-N, make_a(a_2)-N, make_a(a_3)-N)

    print("Potenzmethode: ", speed_test(potenzmethode, a_set))
    print("Nullraum:      ", speed_test(null_space, b_set))

    # print(potenzmethode(v, make_a(a_1)))
    # print(make_a(a_1), make_a(a_2), make_a(a_3), sep="\n\n")
