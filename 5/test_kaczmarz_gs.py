from kaczmarz_gs import kaczmarz_gs

import numpy as np
import random


def generate_matrices(dim: (int, int) = None):
    if dim is None:
        M = random.randint(16, 26)
        N = random.randint(5, 15)
    else:
        M, N = dim

    A = np.random.random(size=(M, N))
    b = np.random.random(size=(M,))
    return A, b


def test_kaczmarz_gs():
    lst = [("N", "average_k / N", "average_k")]
    for N in range(5, 15):
        average_k = 0
        for M in range(16, 27):
            _k = 0
            max_iterations = 100

            for _ in range(max_iterations):
                _k += kaczmarz_gs(*generate_matrices((M, N)), np.zeros(shape=(N,)))[0]

            _k /= max_iterations
            average_k += _k
            print(f"N = {N}, M = {M}:", _k)
        average_k /= 10
        lst.append((N, average_k/N, average_k))
    print("", *lst, sep="\n")


# class TestMatrixDecomposition(TestCase):
#
#     def test_kaczmarz_gs(self):
#

if __name__ == "__main__":
    test_kaczmarz_gs()
