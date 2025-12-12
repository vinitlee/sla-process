import scipy.ndimage as ndi
import cupyx.scipy.ndimage as cndi
import numpy as np
import cupy as cp
import sla_process.core.masking as masking
from line_profiler import profile


def minimize_layers(layers: np.typing.NDArray, margin: int = 5) -> cp.typing.NDArray:
    """
    Takes large input in np and outputs in cp in order to avoid running out of VRAM
    """
    nonzero_x = np.any(layers, axis=(0, 2))
    nonzero_y = np.any(layers, axis=(0, 1))

    x_i = np.where(nonzero_x)[0]
    y_i = np.where(nonzero_y)[0]

    bbox = (slice(None, None), slice(x_i[0], x_i[-1]), slice(y_i[0], y_i[-1]))
    cropped = layers[bbox]

    return cp.pad(
        cp.asarray(cropped, dtype=cp.int16),
        ((0, 1), (margin, margin), (margin, margin)),
    )


def maximize_layers(min_layers: cp.typing.NDArray, full_footprint) -> np.typing.NDArray:

    cp.clip(min_layers, 0, 255, out=min_layers)
    min_layers_np = min_layers.get()
    min_z, min_x, min_y = min_layers_np.shape

    if len(full_footprint) > 2:
        full_footprint = full_footprint[1:3]
    output_footprint = [min_z] + list(full_footprint)
    output = np.zeros(output_footprint, dtype=np.uint8)

    x_pad_total, y_pad_total = np.subtract(
        np.array(full_footprint), np.array([min_x, min_y])
    )
    x_pad = (x_pad_total // 2, x_pad_total - x_pad_total // 2)
    y_pad = (y_pad_total // 2, y_pad_total - y_pad_total // 2)

    output[:min_z, x_pad[0] : x_pad[0] + min_x, y_pad[0] : y_pad[0] + min_y] = (
        min_layers_np
    )

    return output


def collate_layers(
    layers_list: list[np.typing.NDArray],
):
    max_z = np.max([l.shape[0] for l in layers_list])
    for l in layers_list:
        z_diff = max_z - l.shape[0]
        if z_diff:
            l = np.pad(l, ((0, z_diff), (0, 0), (0, 0)))

    rects = [tuple(l.shape[1:]) for l in layers_list]

    pass  # TODO: Implement


def label_layers(layers: cp.typing.NDArray, label: str = "00000", size=100):
    pass  # TODO: Implement


def z_silhouette(layers: cp.typing.NDArray) -> cp.typing.NDArray[bool]:
    return cp.any(layers, axis=0)


def paste(
    source: cp.typing.NDArray, target: cp.typing.NDArray, location=(0, 0)
) -> cp.typing.NDArray:
    pass


def process_array(
    source: cp.typing.NDArray,
    copies: tuple[int, int] = (1, 1),
    processing_fns=[lambda x: x],
    label=None,
):
    """
    Process a source layer stack using each processing function and return a layer stack containing all the processed stacks side by side
    copies defines the output footprint shape
    processing_fns defines the f(in: cp.typing.NDArray) -> cp.typing.NDArray methods that will be used to generate the copies
    if copies[0]*copies[1] and len(processing_fns) are different, limit to the lesser of the two, which will end up with unfilled spots in the grid
    """
    pass


def map_greys(
    source: cp.typing.NDArray,
    in_vals: list[int] | np.typing.NDArray[np.int32],
    out_vals: list[int] | np.typing.NDArray[np.int32],
):
    if len(in_vals) != len(out_vals):
        raise Exception("Length of in_vals and out_vals must be the same")
    if not (0 in in_vals) or not (255 in in_vals):
        raise Exception("in_vals must define both 0 and 255")

    in_vals = np.array(in_vals)
    out_vals = np.array(out_vals)

    sorted_indices = np.argsort(in_vals)
    in_vals = in_vals[sorted_indices]
    out_vals = out_vals[sorted_indices]

    in_vals = np.clip(in_vals, 0, 255)
    out_vals = np.clip(out_vals, 0, 255)

    return cp.interp(
        source,
        cp.array(in_vals),
        cp.array(out_vals),
    )


def project_from_top(
    layers: cp.typing.NDArray, image: cp.typing.NDArray, thickness: int = 1
):
    if image.shape != layers.shape[1:]:
        print("image and layers do not have the same footprint")
    mask = masking.ceil_mask(layers, 2)
    layers[mask] = cp.minimum(layers, image)[mask]


def label(layers: cp.typing.NDArray, size=10, offset_override=(0, 0)):
    # Determine origin
    # Make text
    # Determine slice
    # Project to slice
    pass


def offset_xy(layers: cp.typing.NDArray, size: int, binary: bool = False):
    """
    Docstring for offset_xy

    :param layers: Target layer stack
    :type layers: cp.typing.NDArray
    :param size: size to expand/contract (+/-) from each side. Will change dimensions by size*2
    :type size: int
    :param binary: Flag for binary mode vs grey mode
    :type binary: bool
    """
    # a_size = abs(size)
    # kernel = cp.ones((a_size * 2 + 1, a_size * 2 + 1), bool)

    # if size < 0:
    #     transform = cndi.binary_erosion
    # else:
    #     transform = cndi.binary_dilation
    # for z in range(layers.shape[0]):
    #     print(layers[z, ...].shape)
    #     layers[z, ...] = transform(layers[z, ...], structure=kernel)
    # return layers

    # Cross shaped kernel
    a_size = abs(size)
    kernel = cp.ones((1, 2 * a_size + 1, 2 * a_size + 1), dtype=bool)
    # kernel = np.zeros((1, 2 * a_size + 1, 2 * a_size + 1), dtype=bool)
    # kernel[:, :, a_size] = True
    # kernel[:, a_size, :] = True

    if binary:
        if size == 0:
            return layers
        elif size < 0:
            return cndi.grey_erosion(layers, footprint=kernel)
        else:
            return cndi.grey_dilation(layers, footprint=kernel)
    else:
        if size == 0:
            return layers
        elif size < 0:
            return cndi.binary_erosion(layers, kernel)
        else:
            return cndi.binary_dilation(layers, kernel)
