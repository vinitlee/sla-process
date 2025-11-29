# %%
import scipy.ndimage as ndi
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndi

import sla_process as sla
import matplotlib.pyplot as plt
from pathlib import Path
import importlib

import time

importlib.reload(sla)

# %%
import napari

v = napari.Viewer()  # no napari.run() in notebooks


# %%
def na_refresh(im):
    v.layers.clear()
    v.add_image(im, rgb=False)


def na_replace(im):
    if len(v.layers) < 1:
        na_refresh(im)
    v.layers[0].data = im


# %%
slicer_file = sla.SlicerFile(r"example_data\sl1\sample.sl1")
layers = slicer_file.layers
layers = np.vstack([layers, np.zeros([1] + list(layers.shape[1:]))])
# layers = layers[:, 80:680, 150:500]

na_replace(layers)

# %%
layers_bin = layers > 0

# %%
kernel = cp.array(ndi.generate_binary_structure(3, 1))

dists = cp.zeros(layers_bin.shape, dtype=cp.int16)

start_time = time.perf_counter()
c_layers_bin = cp.array(layers_bin)
eroded = cp.array(layers_bin)
depth = 10
for i in range(depth):
    print(i)
    cndi.binary_erosion(eroded, structure=kernel, output=eroded)
    dists[c_layers_bin & ~eroded & (dists == 0)] = i + 1
end_time = time.perf_counter()
print(end_time - start_time)

start_time = time.perf_counter()
na_replace(dists.get())
end_time = time.perf_counter()
print(end_time - start_time)

# %%
r = 10
kernel = cp.array(np.linalg.norm(np.indices([2 * r + 1] * 3) - r, axis=0) <= r)
processed = cndi.median_filter(c_layers_bin, [r] * 3, mode="constant", cval=0)

na_refresh(processed.get())

# slicer_file.layers = processed.get().astype(np.uint8) * 255
# slicer_file.save(r"example_data\sl1\sample.smoothed.sl1")

# %%
