# %%
import sla_process as sla
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import time

importlib.reload(sla)

import sys

# Dev
import cupy as cp
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndi

from line_profiler import profile
import matplotlib.pyplot as plt

# %%

DEVICE_TEMPLATE_PATH = Path(r"data\templates\PSM\Resione_TH-BJD.phz")
DEVICE_SUFFIX = DEVICE_TEMPLATE_PATH.suffix

layers = None

default_path = Path(r"example_data\sl1\bjd_src.sl1")


# def main():
# global layers
# sample_path = Path(sys.argv[1])
sample_path = default_path
print(f"Loading {sample_path}...")
sample_file = sla.SlicerFile(sample_path.resolve())
print("Loading template...")
output_file = sla.SlicerFile(DEVICE_TEMPLATE_PATH.resolve())
output_path = sample_path.with_suffix(".mod" + DEVICE_SUFFIX)

print("Divvying out layers...")
original_shape = sample_file.layers.shape
layers = sla.minimize_layers(sample_file.layers)

bottom_layers = layers[: sample_file.bottom_layer_count]
reg_layers = layers[sample_file.bottom_layer_count :]
del sample_file

print("Computing masks")
model = sla.mask.model(layers)
skin = sla.mask.skin(layers, thickness=(2, 2, 2))
reg_skin = skin[len(bottom_layers) :]

# print("Applying noise")
# reg_layers[reg_skin] += sla.int_noise(reg_layers[reg_skin].shape, -135, 0)

# %%
print("Applying elephant's foot correction...")

skeleton_dot_kernel = np.array(
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
    dot_kernel: np.typing.NDArray = skeleton_dot_kernel,
) -> np.typing.NDArray:

    # deep erosion
    # skeleton
    # mini erosion
    # dots
    # grey

    layers_cpu: np.typing.NDArray[np.int16] = layers.get()
    layers_cpu_bin = layers_cpu > 0
    processed_layers = np.zeros_like(layers_cpu)

    # deep erosion
    margin_kernel = np.ones(
        (2 * margin + 1, 2 * margin + 1),
        dtype=bool,
    )
    # margin_kernel = np.zeros(
    #     (2 * margin + 1, 2 * margin + 1),
    #     dtype=bool,
    # )
    # margin_kernel[margin, :] = 1
    # margin_kernel[:, margin] = 1

    for z in range(layers_cpu_bin.shape[0]):
        processed_layers[z, ...] = ndi.binary_erosion(
            layers_cpu_bin[z, ...], structure=margin_kernel
        )

    # skeleton
    sk_margin_kernel = np.zeros(
        (2 * skeleton_margin + 1, 2 * skeleton_margin + 1),
        dtype=bool,
    )
    sk_margin_kernel[skeleton_margin, :] = 1
    sk_margin_kernel[:, skeleton_margin] = 1

    for z in range(layers_cpu_bin.shape[0]):
        processed_layers[z, ...] |= (
            ndi.binary_dilation(
                skeletonize(layers_cpu_bin[z, ...]),
                structure=np.ones((thickness, thickness)),
            )
            > 0
        )

    for z in range(layers_cpu_bin.shape[0]):
        margin_mask = (
            ndi.binary_erosion(layers_cpu_bin[z, ...], structure=sk_margin_kernel) > 0
        )
        processed_layers[z, ...] &= margin_mask

    match_slice = (
        slice(0, layers_cpu.shape[0]),
        slice(0, layers_cpu.shape[1]),
        slice(0, layers_cpu.shape[2]),
    )
    tiles = tuple(
        np.divide(layers_cpu.shape, np.concatenate([[1], dot_kernel.shape])).astype(int)
        + 1
    )
    dot_mask = np.tile(dot_kernel, tiles)
    dot_mask = np.maximum(dot_mask * 255, grey)

    processed_layers = (processed_layers > 0) * 255

    processed_layers[layers_cpu_bin] = np.maximum(
        dot_mask[match_slice][layers_cpu_bin], processed_layers[layers_cpu_bin]
    )

    return processed_layers


s = (slice(0, 350), slice(50, 200))
bottom_skeleton = skeleton_foot(
    bottom_layers,
    margin=17,
    skeleton_margin=4,
    thickness=5,
    grey=50,
)
# plt.imshow(bottom_layers[2].get()[s], cmap="gray")
plt.imshow(bottom_skeleton[2][s], cmap="gray")
# %%

bottom_layers[:] = sla.skeleton_foot(bottom_layers, 4)

print("Updating thumbnail...")
heatmap = sla.minimize_thumbnail(sla.layer_heatmap(layers))
output_file.set_thumbnails(heatmap)

print("Restoring original layer size")
layers = sla.maximize_layers(layers, original_shape)

print(f"Saving to {output_path}...")
output_file.layers = layers
output_file.save(output_path)


# if __name__ == "__main__":
#     main()
