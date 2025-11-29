import scipy.ndimage as ndi
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndi
import sla_process.util.kernels as kn
from line_profiler import profile


@profile
def skin(
    layers: np.typing.NDArray,
    thickness: tuple[int, int, int] = (1, 1, 1),
    threshold=1,
    approach="np_erosion",
) -> np.typing.NDArray:

    scratch = np.empty_like(layers, bool)  # reusable temp

    @profile
    def compare_by_slice(source, dest, offset, axis):
        slice_base = [slice(None, None)] * len(source.shape)
        slice_rest = [slice(None, None)] * len(source.shape)
        slice_shift = [slice(None, None)] * len(source.shape)

        if offset >= 0:
            slice_base[axis] = slice(offset, None)
            slice_rest[axis] = slice(None, offset)
            slice_shift[axis] = slice(None, -offset)
        else:
            slice_base[axis] = slice(None, offset)
            slice_rest[axis] = slice(offset, None)
            slice_shift[axis] = slice(-offset, None)

        slice_base = tuple(slice_base)
        slice_rest = tuple(slice_rest)
        slice_shift = tuple(slice_shift)

        np.logical_and(
            source[slice_base],
            np.logical_not(source[slice_shift]),
            out=scratch[slice_base],
        )
        np.logical_or(dest[slice_base], scratch[slice_base], out=dest[slice_base])
        np.logical_or(dest[slice_rest], source[slice_rest], out=dest[slice_rest])

        # dest[slice_base] |= source[slice_base] & ~source[slice_shift]
        # dest[slice_rest] |= source[slice_rest]

    match approach:
        case "slice":
            binary_layers = layers >= threshold
            skin = np.zeros_like(layers, bool)
            for axis in range(len(layers.shape)):
                for direction in [-1, 1]:
                    compare_by_slice(
                        binary_layers, skin, direction * thickness[axis], axis
                    )

            return skin

        case "cp_kernel":
            bin_layer_zeros = cp.array(layers < threshold)
            th_x, th_y, th_z = thickness
            kernel = cp.zeros((1 + 2 * th_x, 1 + 2 * th_y, 1 + 2 * th_z), dtype=bool)
            kernel[th_x, :, :] = True
            kernel[:, th_y, :] = True
            kernel[:, :, th_z] = True

            dilated_zeros = cndi.binary_dilation(bin_layer_zeros, kernel)
            return (~bin_layer_zeros & dilated_zeros).get()
        case "np_erosion":
            binary_layers = layers >= threshold
            kernel = np.zeros((3, 3, 3), dtype=bool)
            kernel[1, :, :] = True
            kernel[:, 1, :] = True
            kernel[:, :, 1] = True

            eroded_zero = (
                ndi.binary_erosion(binary_layers, kernel, iterations=np.max(thickness))
                == 0
            )
            return binary_layers & eroded_zero
        case "np_kernel":
            binary_layers = layers >= threshold
            kernel = np.zeros((3, 3, 3), dtype=bool)
            kernel[1, :, :] = True
            kernel[:, 1, :] = True
            kernel[:, :, 1] = True

            eroded_zero = (
                ndi.binary_erosion(binary_layers, kernel, iterations=np.max(thickness))
                == 0
            )
            return binary_layers & eroded_zero
        case _:
            return np.zeros_like(layers, dtype=bool)


def empty(layers: np.typing.NDArray, threshold=1):
    return layers < threshold


def model(layers: np.typing.NDArray, threshold=1):
    return layers >= threshold


def normal(layers: np.typing.NDArray):
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

    return (c_layers & ~cndi.binary_erosion(c_layers, structure=kernel)).get()


def ceil(layers: np.typing.NDArray, thickess_ceiling: int, thickess_floor: int):
    c_layers = cp.array(layers > 0)
    kernel = cp.array(kn.column_kernel(thickess_ceiling, thickess_floor))

    return (c_layers & ~cndi.binary_erosion(c_layers, structure=kernel)).get()
