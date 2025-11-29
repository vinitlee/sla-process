# %%
import numpy as np
import cupy as cp
import scipy.ndimage as ndi
import cupyx.scipy.ndimage as cndi
import sla_process.core.layers as sla_layers
import cv2

# from line_profiler import profile


def minimize_thumbnail(thumbnail: np.typing.NDArray):
    coords = np.nonzero(thumbnail)
    if coords[0].size == 0:  # No nonzero elements
        print("Looks like an empty array.")
        return thumbnail
    bbox_slices = []
    for dim_coords in coords:
        min_idx = np.min(dim_coords)
        max_idx = np.max(dim_coords)
        bbox_slices.append(slice(min_idx, max_idx + 1))
    bbox_slices = tuple(bbox_slices)

    return thumbnail[bbox_slices]


def maximize_thumbnail(
    min_layers: np.typing.NDArray, target_size: np.typing.ArrayLike
) -> np.typing.NDArray:
    x_pad_total, y_pad_total = np.subtract(target_size, min_layers.shape)
    x_pad = (x_pad_total // 2, x_pad_total - x_pad_total // 2)
    y_pad = (y_pad_total // 2, y_pad_total - y_pad_total // 2)
    return np.pad(min_layers, (x_pad, y_pad))


def fit_thumbnail(
    thumbnail: np.typing.NDArray,
    target_size: np.typing.ArrayLike,
    padding: int = 0,
    rotate: int = 0,
):
    thumbnail = ndi.rotate(thumbnail, rotate)
    scale = np.min(np.divide(np.subtract(target_size, padding), thumbnail.shape))
    scaled = np.array(ndi.zoom(thumbnail, scale))
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    framed = maximize_thumbnail(scaled, target_size)
    return framed


def layer_heatmap(layers: cp.typing.NDArray, color_map=cv2.COLORMAP_BONE):
    heatmap = cp.sum(layers, axis=0, dtype=float).get()
    heatmap /= len(layers)
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, color_map)
    return heatmap_colored


def layer_heightmap(layers: np.typing.NDArray):
    min_layers = sla_layers.minimize_layers(layers)
    pass
