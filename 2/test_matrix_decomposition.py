from unittest import main, TestCase

import numpy as np

from matrix_decomposition import *


class TestMatrixDecomposition(TestCase):
    L: 'np.ndarray' = np.eye(3) - 2 * np.eye(3, k=-1) - 2 * np.eye(3, k=-2)
    R: 'np.ndarray' = np.array([[2, -1, -2],
                                [0, 4, -1],
                                [0, 0, 8]])
    b: 'np.ndarray' = np.array([1., 1., 1.])

    M: 'np.ndarray' = L.dot(R)

    # calculating y
    y: 'np.ndarray' = b.copy()
    y[1] -= y[0] * L[1, 0]
    y[2] -= y[0] * L[2, 0] + y[1] * L[2, 1]

    # calculating x
    x: 'np.ndarray' = y.copy()

    x[2] /= R[2, 2]

    x[1] -= R[1, 2] * x[2]
    x[1] /= R[1, 1]

    x[0] -= R[0, 2] * x[2] + R[0, 1] * x[1]
    x[0] /= R[0, 0]

    # print(l)
    # print(r)
    # print(m)
    #
    # print(b)
    # print(y)
    # print(x)

    def test_test_values(self):
        np.testing.assert_array_almost_equal(self.M.dot(self.x), self.b)
        np.testing.assert_array_almost_equal(self.R.dot(self.x), self.y)

    def test_forward_sub(self):
        np.testing.assert_array_almost_equal(forward_sub(self.L, self.b), self.y)

        # alternative
        np.testing.assert_array_almost_equal(       # L * y = b
            self.L.dot(forward_sub(self.L, self.b)),
            self.b)

    def test_backward_sub(self):
        np.testing.assert_array_almost_equal(backward_sub(self.R, self.y), self.x.astype('float64'))

        # alternative
        np.testing.assert_array_almost_equal(       # M * x = b
            self.M.dot(backward_sub(self.R, forward_sub(self.L, self.b))),
            self.b)

    def test_lu_decomposition(self):
        test_l, test_r = lu_decomposition(self.M)
        np.testing.assert_array_almost_equal(self.L, test_l)
        np.testing.assert_array_almost_equal(self.R, test_r)

    def test_solve_with_lu(self):
        np.testing.assert_array_almost_equal(solve_with_lu(self.M, self.b), self.x)

        # alternative
        np.testing.assert_array_almost_equal(solve_with_lu(self.M, self.b),
                                             backward_sub(self.R, forward_sub(self.L, self.b)))


if __name__ == "__main__":
    main()
