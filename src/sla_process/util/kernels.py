import numpy as np


def spherical_kernel(r: int, dims: int = 3):
    s = 2 * r + 1
    return np.linalg.norm(np.indices([s] * dims) - r, axis=0) <= r


def column_kernel(top: int, bottom: int):
    r = max(top, bottom)
    size = r * 2 + 1
    kernel = np.zeros((size, 3, 3), dtype=bool)
    kernel[:, 1, 1] = (
        [False] * (r - bottom)
        + [True] * bottom
        + [True]
        + [True] * top
        + [False] * (r - top)
    )

    return kernel


def cross_kernel(r: int, dims: int = 3):
    kernel = np.zeros([r * 2 + 1] * dims, dtype=bool)
    for axis in range(kernel.ndim):
        sl = [slice(r, r + 1)] * kernel.ndim
        sl[axis] = slice(None)
        kernel[tuple(sl)] = True
    return kernel
