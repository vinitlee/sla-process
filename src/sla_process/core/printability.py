import numpy as np
import cupy as cp
import scipy.ndimage as ndi
import cupyx.scipy.ndimage as cndi
from skimage.morphology import skeletonize

from line_profiler import profile


def elephants_foot(
    layers: cp.typing.NDArray, pixels: int, num_bottom_layers: int = -1, approach="2d"
) -> cp.typing.NDArray:
    if num_bottom_layers < 0:
        num_bottom_layers = layers.shape[0]
    k = 2 * pixels + 1
    kernel = (cp.linalg.norm(cp.indices((k, k)) - pixels, axis=0) <= pixels)[
        cp.newaxis, :
    ]
    match approach:
        case "2d":
            # 2D approach
            kernel2d = cp.linalg.norm(cp.indices((k, k)) - pixels, axis=0) <= pixels
            for z in range(num_bottom_layers):
                layer = layers[z, ...]
                layer[:] = (
                    cndi.binary_erosion(layer > 0, structure=kernel2d).astype(
                        layers.dtype
                    )
                    * 255
                )
        case "3d":
            bottom_layers = layers[:num_bottom_layers]
            bottom_layers[:] = cndi.grey_erosion(
                cp.array(bottom_layers), structure=cp.array(kernel)
            )

    return layers


def beveled_elephants_foot(
    bottom_layers: np.typing.NDArray, pixels_bottom: int = 6, pixels_top: int = 4
):
    pass


SKEL_DOT_KERNEL = np.array(
    [
        [1, 0],
        [0, 0],
        [0, 1],
        [0, 0],
    ]
)


def skeleton_foot(
    layers: cp.typing.NDArray[cp.int16],
    margin: int = 35,
    skeleton_margin: int = 0,
    thickness: int = 1,
    grey: int = 128,
    dot_kernel: np.typing.NDArray = SKEL_DOT_KERNEL,
) -> cp.typing.NDArray:

    layers_cpu_b = layers.get() > 0
    processed_layers = np.zeros_like(layers_cpu_b, dtype=layers.dtype)

    dst = cp.zeros_like(processed_layers, np.float32)
    for z in range(layers_cpu_b.shape[0]):
        dst[z, ...] = cndi.distance_transform_edt(
            image=layers[z, ...],
            return_distances=True,
            return_indices=False,
            float64_distances=False,
        )
    dst = dst.get()

    # deep erosion
    processed_layers[:] = dst > margin

    # skeleton
    for z in range(layers_cpu_b.shape[0]):
        processed_layers[z, ...] |= (
            ndi.binary_dilation(
                skeletonize(layers_cpu_b[z, ...]),
                structure=np.ones((thickness, thickness)),
            )
            > 0
        )

    # keepout margin
    # TODO: Handle really thin sections
    # Disabling skeleton margin at the moment, but optimally this is a combination of approaches.
    # processed_layers &= dst > skeleton_margin

    # dots & grey
    match_slice = (
        slice(0, layers_cpu_b.shape[0]),
        slice(0, layers_cpu_b.shape[1]),
        slice(0, layers_cpu_b.shape[2]),
    )
    tiles = tuple(
        np.divide(layers_cpu_b.shape, np.concatenate([[1], dot_kernel.shape])).astype(
            int
        )
        + 1
    )
    dot_mask = np.tile(dot_kernel, tiles)[match_slice]
    dot_mask = np.maximum((dot_mask & (dst > skeleton_margin)) * 255, grey)

    processed_layers = (processed_layers > 0) * 255

    processed_layers[layers_cpu_b] = np.maximum(
        dot_mask[layers_cpu_b], processed_layers[layers_cpu_b]
    )

    return cp.array(processed_layers)


def remove_islands(layers: np.typing.NDArray, iterations: int = 1):
    pass


def remove_orphans(layers: np.typing.NDArray, iterations: int = 1):
    pass
