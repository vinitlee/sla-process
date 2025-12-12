import scipy.ndimage as ndi
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndi
import sla_process.util.kernels as kn
import sla_process.util.tools as tl
from line_profiler import profile


def skin(
    layers: cp.typing.NDArray,
    thickness: tuple[int, int, int] = (1, 1, 1),
    threshold=1,
) -> cp.typing.NDArray:

    th_x, th_y, th_z = thickness
    kernel = cp.zeros((1 + 2 * th_x, 1 + 2 * th_y, 1 + 2 * th_z), dtype=bool)
    kernel[th_x, :, :] = True
    kernel[:, th_y, :] = True
    kernel[:, :, th_z] = True

    bin_layer_zeros = layers < threshold
    dilated_zeros = cndi.binary_dilation(bin_layer_zeros, kernel)
    return ~bin_layer_zeros & dilated_zeros


def skin_lite(
    layers: cp.typing.NDArray,
    thickness: tuple[int, int, int] = (1, 1, 1),
    threshold: int | float = 1,
) -> cp.typing.NDArray:
    """
    Approximate 'skin' mask, computed on CPU.

    A voxel is skin if:
      - layers[z, x, y] >= threshold (solid), and
      - at least one of its 6 neighbors at distance `thickness` along X/Y/Z is < threshold,
        with everything outside the volume treated as 0 (air).

    This uses exactly two comparisons per direction (±th along each axis).
    """

    # Binary solid mask on host
    bin_layer = (layers >= threshold).get().astype(bool)  # (Z, X, Y)
    Z, X, Y = bin_layer.shape
    th_x, th_y, th_z = thickness

    zeros = ~bin_layer  # True where air
    zero_neighbor = np.zeros_like(bin_layer, dtype=bool)

    # X-direction neighbors (axis 1)
    if th_x > 0:
        # +X neighbor (inside domain)
        if th_x < X:
            zero_neighbor[:, th_x:, :] |= zeros[:, :-th_x, :]
            zero_neighbor[:, :-th_x, :] |= zeros[:, th_x:, :]

    # Y-direction neighbors (axis 2)
    if th_y > 0:
        # +Y neighbor (inside domain)
        if th_y < Y:
            zero_neighbor[:, :, th_y:] |= zeros[:, :, :-th_y]
            zero_neighbor[:, :, :-th_y] |= zeros[:, :, th_y:]

    # Z-direction neighbors (axis 0)
    if th_z > 0:
        # +Z neighbor (inside domain)
        if th_z < Z:
            zero_neighbor[th_z:, :, :] |= zeros[:-th_z, :, :]
            zero_neighbor[:-th_z, :, :] |= zeros[th_z:, :, :]

    # Handle edges as if outside is always air (zeros == True)
    # Any solid voxel whose ±th neighbor would be out-of-bounds
    # should be treated as having a zero neighbor.

    # X edges
    if th_x > 0:
        # Left edge: x < th_x  → neighbor at x - th_x is outside
        x_left = slice(0, min(th_x, X))
        zero_neighbor[:, x_left, :] |= bin_layer[:, x_left, :]

        # Right edge: x > X-1 - th_x → neighbor at x + th_x is outside
        x_right_start = max(0, X - th_x)
        if x_right_start < X:
            x_right = slice(x_right_start, X)
            zero_neighbor[:, x_right, :] |= bin_layer[:, x_right, :]

    # Y edges
    if th_y > 0:
        # Front edge: y < th_y  → neighbor at y - th_y is outside
        y_front = slice(0, min(th_y, Y))
        zero_neighbor[:, :, y_front] |= bin_layer[:, :, y_front]

        # Back edge: y > Y-1 - th_y → neighbor at y + th_y is outside
        y_back_start = max(0, Y - th_y)
        if y_back_start < Y:
            y_back = slice(y_back_start, Y)
            zero_neighbor[:, :, y_back] |= bin_layer[:, :, y_back]

    # Z edges
    if th_z > 0:
        # Bottom edge: z < th_z  → neighbor at z - th_z is outside
        z_bottom = slice(0, min(th_z, Z))
        zero_neighbor[z_bottom, :, :] |= bin_layer[z_bottom, :, :]

        # Top edge: z > Z-1 - th_z → neighbor at z + th_z is outside
        z_top_start = max(0, Z - th_z)
        if z_top_start < Z:
            z_top = slice(z_top_start, Z)
            zero_neighbor[z_top, :, :] |= bin_layer[z_top, :, :]

    # Final skin mask
    skin = bin_layer & zero_neighbor
    return cp.array(skin)


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
    return wall(c_layers, thickess_wall) | ceil_floor(
        c_layers, thickness_ceil, thickness_floor
    )


def wall(layers: np.typing.NDArray | cp.ndarray, thickness: int):
    c_layers = cp.array(layers > 0)
    kernel = cp.array(kn.spherical_kernel(thickness, 2)[np.newaxis, ...])

    return c_layers & ~cndi.binary_erosion(c_layers, structure=kernel)


def ceil_floor(layers: np.typing.NDArray, thickess_ceiling: int, thickess_floor: int):
    c_layers = cp.array(layers > 0)
    kernel = cp.array(kn.column_kernel(thickess_ceiling, thickess_floor))

    return c_layers & ~cndi.binary_erosion(c_layers, structure=kernel)


def ceil_mask(layers: cp.typing.NDArray, thickness: int):
    layers_bin = layers == 0
    kernel = cp.zeros((2 * thickness + 1, 2 * thickness + 1, 2 * thickness + 1))
    kernel[0 : thickness + 1, thickness, thickness] = True

    return ~layers_bin & cndi.binary_dilation(layers_bin, structure=kernel)
