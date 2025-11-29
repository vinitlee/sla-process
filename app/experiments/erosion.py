# %%
import sla_process as sla
import scipy.ndimage as ndi
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as cndi
from pathlib import Path
import sla_process.util.kernels as kn

import napari

# %%
v = napari.Viewer()

file_path = Path(r"example_data\sl1\sample.sl1")
slicer_file = sla.SlicerFile(file_path)

# %%
layers = slicer_file.layers
layers = np.pad(layers, [(0, 1), (0, 0), (0, 0)])
c_layers = cp.array(layers)
c_layers_bin = c_layers > 0

kernel = cp.array(kn.spherical_kernel(1, 3))
# kernel = cp.array(kn.cross_kernel(1, 3))
dists = cp.full(c_layers_bin.shape, fill_value=-1, dtype=int)
eroded = c_layers_bin.copy()
thickness = 5
for l in range(thickness):
    cndi.binary_erosion(eroded, kernel, output=eroded)
    dists[(dists == -1) & (c_layers_bin & ~eroded)] = l

v.add_image(dists.get())

# %%
# v.layers.clear()
coords = cp.indices((3, 3, 3), dtype=cp.float32)  # shape (3,3,3,3)
# center at (1,1,1)
dist = cp.sqrt((coords[0] - 1) ** 2 + (coords[1] - 1) ** 2 + (coords[2] - 1) ** 2)
sphere19 = dist <= 1.5  # bool footprint

r = 1
kernel = np.linalg.norm(np.indices([r * 2 + 1] * 3, dtype=float) - r, axis=0)
kernel = np.where(kernel < kernel.max(), 0, -255)
kernel = cp.array(kernel.astype(np.int16))

# kernel = np.where(kn.spherical_kernel(2, 3), 0, -32)

c_layers = cp.array(layers)
c_noise = cp.zeros_like(c_layers)
noise_percentage = 0.3
noise_src = sla.noise(c_noise[dists == 0].shape, 0, 255)
noise_src -= 255 * (1 - noise_percentage)
noise_src *= 255 / noise_src.max()
noise_src[noise_src < 0] = 0
c_noise[dists == 0] = noise_src

depth = 5
for i in range(depth):
    if i != 0:
        c_noise[:] = cndi.grey_dilation(
            c_noise, structure=kernel, mode="constant", cval=0
        )
    # c_noise[dists == -1] = 0
    c_layers[dists == i] = c_noise[dists == i]

noise_range = [200, 255]
c_layers[(dists <= depth) & (dists >= 0)] = cp.interp(
    c_layers[(dists <= depth) & (dists >= 0)],
    cp.array([0, 255]),
    cp.array(noise_range),
)

if len(v.layers) == 0:
    v.add_image(c_layers.get())
else:
    v.layers[0].data = c_layers.get()


# %%
vol = cp.zeros((5, 5, 5), dtype=cp.uint8)
vol[0, 2, 2] = 1
# 3D footprint
foot = cp.ones((3, 3, 3))

res = cp.zeros_like(vol)
cndi.grey_dilation(vol, footprint=foot, output=res)

print(res[:, 2, 2])
