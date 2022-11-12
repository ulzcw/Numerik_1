import numpy as np


if __name__ == "__main__":
    b = np.array([1, 2, 3])
    A = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [4, 5, 6]])

    b1 = np.concatenate([np.ones(2, dtype=int), -1 * np.ones(2, dtype=int)])
    A1 = 5*np.eye(3, dtype=int) + 4*np.eye(3, k=1, dtype=int) + 3*np.eye(3, k=-1, dtype=int)

    b2 = np.zeros(3, dtype=int)
    A2 = np.ones((4, 2), dtype=int)

    print(b, "= b")
    print(A, "= A\n")

    print("b)", b1, A1, b2, A2, sep="\n\n")

    print("c)", b1[0], b2[-1], A1[0, :], A2[:, -1], sep="\n\n")
