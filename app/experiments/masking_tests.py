# %%
import sla_process as sla
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import cupy as cp
import cupyx.scipy.ndimage as cndi
import sla_process.util.kernels as kn

import napari

# %%
v = napari.Viewer()

file_path = Path(r"example_data\sl1\sample.sl1")
slicer_file = sla.SlicerFile(file_path)

# %%
layers = slicer_file.layers[:]

layers_bin = layers > 0

# %%
v.layers.clear()
walls = sla.mask.wall(layers, 2)
ceils = sla.mask.ceil_floor(layers, 2, 0)
floors = sla.mask.ceil_floor(layers, 0, 3)
v.add_image(walls, colormap="gray", blending="additive")
v.add_image(ceils, colormap="bop orange", blending="additive")
v.add_image(floors, colormap="cyan", blending="additive")
