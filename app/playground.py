# %%
import sla_process as sla
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

importlib.reload(sla)

import sys

# Dev
import time
import cupy as cp
import numpy as np
from skimage.morphology import skeletonize
import scipy.ndimage as ndi
import cupyx.scipy.ndimage as cndi
import matplotlib.pyplot as plt

from line_profiler import profile

# %%

DEVICE_TEMPLATE_PATH = Path(r"data\templates\PSM\Resione_TH-BJD.phz")
DEVICE_SUFFIX = DEVICE_TEMPLATE_PATH.suffix

layers = None

# sample_path = Path(sys.argv[1])
sample_path = Path(r"example_data\sl1\sm_test.sl1")

print(f"Loading {sample_path}...")
sample_file = sla.SlicerFile(sample_path.resolve())
print("Loading template...")
output_file = sla.SlicerFile(DEVICE_TEMPLATE_PATH.resolve())
output_path = sample_path.with_suffix(".bjd" + DEVICE_SUFFIX)

# %%

print("Divvying out layers...")
original_shape = sample_file.layers.shape
layers = sla.minimize_layers(sample_file.layers)

bottom_layers = layers[: sample_file.bottom_layer_count]
reg_layers = layers[sample_file.bottom_layer_count :]
del sample_file

# %%
skin = sla.mask.skin_lite(layers, (2, 2, 2), 1)
plt.imshow(skin[-4, ...])

# %%


def pattern(tile: np.typing.NDArray, target_shape):
    num_tiles = list((np.divide(target_shape, tile.shape).astype(np.int16) + 1))
    full = np.tile(tile, num_tiles)

    sl = tuple([slice(None, sh) for sh in target_shape])
    return full[sl]


def ceil_mask(layers: cp.typing.NDArray, thickness: int):
    layers_bin = layers == 0
    kernel = cp.zeros((2 * thickness + 1, 2 * thickness + 1, 2 * thickness + 1))
    kernel[0 : thickness + 1, thickness, thickness] = True

    return ~layers_bin & cndi.binary_dilation(layers_bin, structure=kernel)


def project_from_top(
    layers: cp.typing.NDArray, image: cp.typing.NDArray, thickness: int = 1
):
    if image.shape != layers.shape[1:]:
        print("image and layers do not have the same footprint")
    mask = sla.mask.ceil_mask(layers, 2)
    layers[mask] = cp.minimum(layers, image)[mask]


# mask = ceil_mask(reg_layers, 2)
tile_1 = (
    np.array(
        [
            [1, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    * 255
).astype(np.uint16)
pattern_1 = cp.array(pattern(tile_1, layers.shape[1:]))

# cp.putmask(reg_layers, mask, pattern_1)

project_from_top(layers, pattern_1, 2)

plt.imshow(layers[20, 50:200, 80:120].get())
# %%

# print("Computing masks")
# model = sla.mask.model(layers)
# skin = sla.mask.skin(layers, thickness=(2, 2, 2))
# reg_skin = skin[len(bottom_layers) :]

print("Applying elephant's foot correction...")
bottom_layers[:] = sla.skeleton_foot(
    bottom_layers,
    margin=19,
    skeleton_margin=4,
    thickness=5,
    grey=40,
)

print("Applying noise")
reg_layers[:] = sla.noisy_greys(reg_layers[:], 120)

print("Mapping greys")
reg_layers[:] = sla.map_greys(reg_layers, [0, 1, 255], [0, 160, 255])

print("Updating thumbnail...")
heatmap = sla.minimize_thumbnail(sla.layer_heatmap(layers))
output_file.set_thumbnails(heatmap)

print("Restoring original layer size")
layers = sla.maximize_layers(layers, original_shape)

print("Setting custom params")
# TODO: Check that WTBC/LoD is set correctly
output_file.print_params.update(
    {
        "WaitTimeBeforeCure": 3,
        "LiftHeight": 8,
        "LiftSpeed": 60,
        "ExposureTime": 3.1,
    }
)

output_file.layers = layers

# temp_output_path = output_path.with_stem(output_path.stem + f"_v{g1}")
# print(f"Saving to {temp_output_path}...")
# output_file.save(temp_output_path)

print(f"Saving to {output_path}...")
output_file.save(output_path)

# Exposure series
# import numpy as np

# output_file.layers = layers

# exposure_times = np.round(np.linspace(2.8, 3.5, num=5), 2)
# for e in exposure_times:
#     output_file.print_params["ExposureTime"] = e
#     sub_output_path = output_path.with_stem(output_path.stem + f"_t{e}")
#     print(f"Saving time {e} to {sub_output_path}...")
#     output_file.save(sub_output_path)
