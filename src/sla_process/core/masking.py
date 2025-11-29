import scipy.ndimage as ndi
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndi
import sla_process.util.kernels as kn
from line_profiler import profile


def skin(
    layers: cp.typing.NDArray,
    thickness: tuple[int, int, int] = (1, 1, 1),
    threshold=1,
) -> cp.typing.NDArray:

    bin_layer_zeros = cp.array(layers < threshold)
    th_x, th_y, th_z = thickness
    kernel = cp.zeros((1 + 2 * th_x, 1 + 2 * th_y, 1 + 2 * th_z), dtype=bool)
    kernel[th_x, :, :] = True
    kernel[:, th_y, :] = True
    kernel[:, :, th_z] = True

    dilated_zeros = cndi.binary_dilation(bin_layer_zeros, kernel)
    return ~bin_layer_zeros & dilated_zeros


def empty(layers: cp.typing.NDArray, threshold=1):
    return layers < threshold


def model(layers: cp.typing.NDArray, threshold=1):
    return layers >= threshold


def normal(layers: cp.typing.NDArray):
    pass


def wall_and_ceil(
    layers: np.typing.NDArray,
    thickess_wall: int,
    thickness_ceil: int,
    thickness_floor: int = 0,
):
    """
    Return a boolean combination of 2D walls and ceilings and floors.
    A type of skin that guarantees a certain number of walls
    """
    c_layers = cp.array(layers > 0)
    return wall(c_layers, thickess_wall) | ceil(
        c_layers, thickness_ceil, thickness_floor
    )


def wall(layers: np.typing.NDArray | cp.ndarray, thickness: int):
    c_layers = cp.array(layers > 0)
    kernel = cp.array(kn.spherical_kernel(thickness, 2)[np.newaxis, ...])

    return c_layers & ~cndi.binary_erosion(c_layers, structure=kernel)


def ceil(layers: np.typing.NDArray, thickess_ceiling: int, thickess_floor: int):
    c_layers = cp.array(layers > 0)
    kernel = cp.array(kn.column_kernel(thickess_ceiling, thickess_floor))

    return c_layers & ~cndi.binary_erosion(c_layers, structure=kernel)
