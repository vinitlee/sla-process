import numpy as np
import cupy as cp
import scipy.ndimage as ndi
import cupyx.scipy.ndimage as cndi
from skimage.morphology import skeletonize

from line_profiler import profile


def elephants_foot(
    layers: np.typing.NDArray, pixels: int, num_bottom_layers: int = -1, approach="2d"
) -> np.typing.NDArray:
    if num_bottom_layers < 0:
        num_bottom_layers = layers.shape[0]
    k = 2 * pixels + 1
    kernel = (np.linalg.norm(np.indices((k, k)) - pixels, axis=0) <= pixels)[
        np.newaxis, :
    ]
    match approach:
        case "2d":
            # 2D approach
            kernel2d = np.linalg.norm(np.indices((k, k)) - pixels, axis=0) <= pixels
            for z in range(num_bottom_layers):
                layer = layers[z, ...]
                layer[:] = (
                    ndi.binary_erosion(layer > 0, structure=kernel2d).astype(
                        layers.dtype
                    )
                    * 255
                )
        case "3d":
            bottom_layers = layers[:num_bottom_layers]
            # bottom_layers[:] = ndi.grey_erosion(bottom_layers, structure=kernel)
            bottom_layers[:] = cndi.grey_erosion(
                cp.array(bottom_layers), structure=cp.array(kernel)
            ).get()

    return layers


def beveled_elephants_foot(
    bottom_layers: np.typing.NDArray, pixels_bottom: int = 6, pixels_top: int = 4
):
    pass


def skeleton_foot(
    layers: np.typing.NDArray,
    margin: int,
    thickness: int = 2,
    num_bottom_layers: int | None = None,
) -> np.typing.NDArray:
    if num_bottom_layers is not None:
        bottom_layers = layers[:num_bottom_layers]
    else:
        bottom_layers = layers[:]

    bottom_layers[:] = skeletonize(bottom_layers > 0)

    return bottom_layers


def remove_islands(layers: np.typing.NDArray, iterations: int = 1):
    pass


def remove_orphans(layers: np.typing.NDArray, iterations: int = 1):
    pass
