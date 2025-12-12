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


DEVICE_TEMPLATE_PATH = Path(r"data\templates\PSM\Resione_TH-BJD.phz")
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
    output_path = sample_path.with_suffix(".bjd" + DEVICE_SUFFIX)

    print("Divvying out layers...")
    original_shape = sample_file.layers.shape
    layers = sla.minimize_layers(sample_file.layers)

    bottom_layers = layers[: sample_file.bottom_layer_count]
    reg_layers = layers[sample_file.bottom_layer_count :]
    del sample_file

    print("Applying elephant's foot correction...")
    bottom_layers[:] = sla.skeleton_foot(
        bottom_layers,
        margin=19,
        skeleton_margin=4,
        thickness=5,
        grey=40,
    )

    # print("Applying noise")
    # reg_layers[:] = sla.noisy_greys(reg_layers[:], 120)

    # print("Mapping greys")
    # reg_layers[:] = sla.map_greys(reg_layers, [0, 1, 255], [0, 160, 255])

    print("Thresholding")
    reg_layers[:] = sla.map_greys(reg_layers, [0, 127, 128, 255], [0, 0, 255, 255])

    print("X/Y Contraction")
    reg_layers[:] = sla.offset_xy(reg_layers[:], -1, binary=True)

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
            "ExposureTime": 3.2,
        }
    )

    output_file.layers = layers

    print(f"Saving to {output_path}...")
    output_file.save(output_path)


if __name__ == "__main__":
    main()
