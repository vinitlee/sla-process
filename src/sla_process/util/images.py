import numpy as np
import cupy as cp


def pattern(tile: np.typing.NDArray, target_shape):
    num_tiles = list((np.divide(target_shape, tile.shape).astype(np.int16) + 1))
    full = np.tile(tile, num_tiles)

    sl = tuple([slice(None, sh) for sh in target_shape])
    return full[sl]
