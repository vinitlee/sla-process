import scipy.ndimage as ndi
import numpy as np
import cupy as cp


def minimize_layers(layers: np.typing.NDArray, margin: int = 5) -> np.typing.NDArray:
    if not np.any(layers):
        print("Warning: layers are blank")
        return layers

    nonzero_z = np.any(layers, axis=(1, 2))
    nonzero_x = np.any(layers, axis=(0, 2))
    nonzero_y = np.any(layers, axis=(0, 1))

    z_i = np.where(nonzero_z)[0]
    x_i = np.where(nonzero_x)[0]
    y_i = np.where(nonzero_y)[0]

    bbox = (slice(z_i[0], z_i[-1]), slice(x_i[0], x_i[-1]), slice(y_i[0], y_i[-1]))

    return np.pad(layers[bbox], ((0, 0), (margin, margin), (margin, margin)))


def maximize_layers(
    min_layers: np.typing.NDArray, full_footprint: np.typing.ArrayLike
) -> np.typing.NDArray:

    full_footprint = np.array(full_footprint)
    if len(full_footprint) > 2:
        full_footprint = full_footprint[1:3]

    x_pad_total, y_pad_total = np.subtract(full_footprint, min_layers.shape[1:])
    x_pad = (x_pad_total // 2, x_pad_total - x_pad_total // 2)
    y_pad = (y_pad_total // 2, y_pad_total - y_pad_total // 2)
    z_pad = (0, 0)

    return np.pad(min_layers, (z_pad, x_pad, y_pad))


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
