import scipy.ndimage as ndi
import numpy as np
import cupy as cp
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


def label_layers(layers: np.typing.NDArray, label: str = "00000", size=100):
    pass  # TODO: Implement
