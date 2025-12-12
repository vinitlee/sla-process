import numpy as np
import cupy as cp


def pingpong(a: np.typing.NDArray, a_min, a_max):
    # TODO: handle more than one ping
    a[a < a_min] = a_min + (a_min - a[a < a_min])
    a[a > a_max] = a_max - (a[a > a_max] - a_max)

    return a
