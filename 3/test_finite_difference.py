import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from finite_difference import *
from math import exp


def f(x):
    return x * exp(x)


def d_f(x):
    return math.e**x + f(x)
    # return exp(x) + f(x)


def log_values(max_exp):
    for i in range(1, max_exp + 1):
        yield 10 ** (-i)


def error_plot():
    # h = log_values(10)
    # h2 = np.linspace(10**-1, 10**-10, 10000)
    # h = tuple(10**(-i) for i in range(1, 11))
    h = np.asarray(tuple(10**(-i) for i in range(10, 0, -1)))
    print("h =", h)
    x = 2 * np.ones(h.shape)

    x_fd = lambda _h: abs(d_f(x) - finite_difference_forward(f, x, _h))
    x_zd = lambda _h: abs(d_f(x) - finite_difference_central(f, x, _h))

    # plt.loglog(h, tuple(map(x_fd, h)))
    # plt.loglog(h, tuple(map(x_zd, h)))
    plt.loglog(h, x_zd(h))
    plt.show()


if __name__ == "__main__":
    error_plot()


