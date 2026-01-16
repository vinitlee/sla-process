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

import cv2

from line_profiler import profile

print("WARNING: THIS IS NOT DONE")

DEVICE_TEMPLATE_PATH = Path("data/templates") / "PSM" / "Resione_K+.phz"
CONFIG_SUFFIX = ".kp"
DEVICE_SUFFIX = DEVICE_TEMPLATE_PATH.suffix

layers = None


def main():
    global layers
    sample_path = Path(sys.argv[1])
    # sample_path = Path(r"example_data\sl1\bjd_src.sl1")

    print(f"Loading {sample_path}...")
    sample_file = sla.SlicerFile(sample_path.resolve())
    print("Loading template...")
    output_file = sla.SlicerFile(DEVICE_TEMPLATE_PATH.resolve())
    output_path = sample_path.with_suffix(CONFIG_SUFFIX + DEVICE_SUFFIX)

    print("Divvying out layers...")
    original_shape = sample_file.layers.shape
    layers = sla.minimize_layers(sample_file.layers)

    bottom_layers = layers[: sample_file.bottom_layer_count]
    reg_layers = layers[sample_file.bottom_layer_count :]
    reg_skin_xy = sla.mask.skin(layers, thickness=(0, 1, 1), threshold=1)[
        sample_file.bottom_layer_count :
    ]
    reg_skin_z = sla.mask.skin(layers, thickness=(1, 0, 0), threshold=1)[
        sample_file.bottom_layer_count :
    ]
    del sample_file

    print("Applying elephant's foot correction...")
    bottom_layers[:] = sla.elephants_foot(
        bottom_layers,
        pixels=2,
    )

    print("Applying noise")
    # Noise to grey regions
    # reg_layers[:] = sla.noisy_greys(reg_layers[:], 32, 0, 248)
    # Noise to all walls
    # reg_layers[reg_skin_xy] = sla.noisy(reg_layers[reg_skin_xy], 16)
    # Noise to all ceil/floor
    # reg_layers[reg_skin_z] = sla.noisy(reg_layers[reg_skin_z], 48)

    print("Mapping greys")
    # Tonemap walls
    reg_layers[reg_skin_xy] = sla.map_greys(
        reg_layers[reg_skin_xy], [0, 1, 255], [0, 50, 210]
    )
    # Tonemap ceil/floor
    # reg_layers[reg_skin_z] = sla.map_greys(
    #     reg_layers[reg_skin_z], [0, 1, 255], [0, 32, 242]
    # )

    print("Updating thumbnail...")
    heatmap = sla.minimize_thumbnail(sla.layer_heatmap(layers, cv2.COLORMAP_INFERNO))
    output_file.set_thumbnails(heatmap)

    print("Restoring original layer size")
    layers = sla.maximize_layers(layers, original_shape)

    print("Setting custom params")
    # TODO: Check that WTBC/LoD is set correctly
    output_file.print_params.update(
        {
            "WaitTimeBeforeCure": 2,
            "BottomWaitTimeBeforeCure": 4,
        }
    )

    output_file.layers = layers

    print(f"Saving to {output_path}...")
    output_file.save(output_path)


if __name__ == "__main__":
    main()

# %%
