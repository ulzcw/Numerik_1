from unittest import main, TestCase
from matrix_decomposition import *


class TestMatrixDecomposition(TestCase):
    l = np.eye(3) - 2*np.eye(3, k=-1) - 2*np.eye(3, k=-2)
    r = np.array([[2, -1, -2],
                  [0, 4, -1],
                  [0, 0, 8]])
    m = l.dot(r)

    def test_forward_sub(self):
        np.testing.assert_array_almost_equal()

    def test_backward_sub(self):
        pass

    def test_lu_decomposition(self):
        test_l, test_r = lu_decomposition(self.m)
        np.testing.assert_array_almost_equal(self.l, test_l)
        np.testing.assert_array_almost_equal(self.r, test_r)

    def test_solve_with_lu(self):
        pass


if __name__ == "__main__":
    main()
