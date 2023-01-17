from unittest import main, TestCase
from typing import Iterable
from kaczmarz_gs import kaczmarz_gs

import numpy as np
import random


def gen(dim: (int, int) = None):
    if dim is None:
        M = random.randint(16, 26)
        N = random.randint(5, 15)
    else:
        M, N = dim

    A = np.random.random(size=(M, N))
    b = np.random.random(size=(M,))
    return A, b


def test_kaczmarz_gs():
    for M in range(16, 27):
        for N in range(5, 15):
            print(f"({M}, {N}):", kaczmarz_gs(*gen((M, N)), np.zeros(shape=(N,)))[0])


# class TestMatrixDecomposition(TestCase):
#
#     def test_kaczmarz_gs(self):
#

if __name__ == "__main__":
    test_kaczmarz_gs()
