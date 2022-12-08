from typing import Callable

import numpy as np


def finite_difference_forward(f: Callable, x, h):
    return (f(x + h) - f(x)) / h


def finite_difference_central(f: Callable, x, h):
    print("x =", x)
    print("h =", h)
    return (f(x + h) - f(x - h)) / 2*h
