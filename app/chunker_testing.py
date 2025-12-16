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

import tqdm

# %%

DEVICE_TEMPLATE_PATH = Path(r"data\templates\PSM\Resione_TH-BJD.phz")
DEVICE_SUFFIX = DEVICE_TEMPLATE_PATH.suffix

# layers = None

# sample_path = Path(sys.argv[1])
# sample_path = Path(r"example_data\sl1\sm_test.sl1")
sample_path = Path(r"example_data\sl1\big_sample.sl1")

print(f"Loading {sample_path}...")
sample_file = sla.SlicerFile(sample_path.resolve())
print("Loading template...")
output_file = sla.SlicerFile(DEVICE_TEMPLATE_PATH.resolve())
output_path = sample_path.with_suffix(".bjd" + DEVICE_SUFFIX)

# %%

print("Divvying out layers...")
original_shape = sample_file.layers.shape
# layers = sla.minimize_layers(sample_file.layers).get()
layers = sample_file.layers


# %%
def n_chunks(
    layers: np.typing.NDArray | cp.typing.NDArray,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
):
    z, x, y = layers.shape
    cz, cx, cy = np.ceil(np.divide([z, x, y], chunk_size)).astype(int)
    total = cz * cx * cy

    return total


@profile
def chunk_generator(
    layers: np.typing.NDArray | cp.typing.NDArray,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    margin: int = 8,
    fill_value=0,
):
    match type(layers):
        case np.ndarray:
            z, x, y = layers.shape
            cz, cx, cy = np.ceil(np.divide([z, x, y], chunk_size)).astype(int)

            chunk_indices = np.ndindex((cz, cx, cy))

            chunk_buf = np.empty(np.add(chunk_size, 2 * margin), dtype=layers.dtype)
        case cp.ndarray:
            z, x, y = layers.shape
            cz, cx, cy = np.ceil(np.divide([z, x, y], chunk_size)).astype(int)

            chunk_indices = np.ndindex((cz, cx, cy))

            chunk_buf = cp.empty(np.add(chunk_size, 2 * margin), dtype=cp.int16)
        case _:
            raise Exception(f"layers not of supported type: {type(layers)}")
    for czi, cxi, cyi in chunk_indices:
        core_start = np.multiply([czi, cxi, cyi], chunk_size)
        core_z = min(chunk_size[0], z - core_start[0])
        core_x = min(chunk_size[1], x - core_start[1])
        core_y = min(chunk_size[2], y - core_start[2])
        core_slice = tuple(
            [slice(margin, margin + s) for s in [core_z, core_x, core_y]]
        )
        chunk_slice = (
            slice(core_start[0], core_start[0] + core_z),
            slice(core_start[1], core_start[1] + core_x),
            slice(core_start[2], core_start[2] + core_y),
        )
        chunk_buf.fill(fill_value)

        slice_start = core_start - margin
        slice_start_p = np.maximum(0, slice_start)
        slice_offset = slice_start_p - slice_start
        valid_slice = layers[
            slice_start_p[0] : slice_start[0] + core_z + 2 * margin,
            slice_start_p[1] : slice_start[1] + core_x + 2 * margin,
            slice_start_p[2] : slice_start[2] + core_y + 2 * margin,
        ]
        vs_z, vs_x, vs_y = valid_slice.shape
        chunk_buf[
            slice_offset[0] : slice_offset[0] + vs_z,
            slice_offset[1] : slice_offset[1] + vs_x,
            slice_offset[2] : slice_offset[2] + vs_y,
        ] = valid_slice
        buffer_data_slice = (
            slice(core_z + 2 * margin),
            slice(core_x + 2 * margin),
            slice(core_y + 2 * margin),
        )
        # Also put margins in, defaulting to fill
        yield chunk_buf[buffer_data_slice], chunk_slice, core_slice


@profile
def gpu_chunk_generator(
    layers_gpu: cp.typing.NDArray,
    chunk_size: tuple[int, int, int] = (128, 128, 128),
    margin: int = 8,
    fill_value=0,
):
    """
    layers_gpu should optimally be dtype=cp.uint8 to save VRAM
    """
    z, x, y = layers_gpu.shape
    cz, cx, cy = np.ceil(np.divide([z, x, y], chunk_size)).astype(int)

    total = cz * cx * cy

    chunk_indices = np.ndindex((cz, cx, cy))

    chunk_buf = cp.empty(np.add(chunk_size, 2 * margin), dtype=cp.int16)
    for czi, cxi, cyi in chunk_indices:
        core_start = np.multiply([czi, cxi, cyi], chunk_size)
        core_z = min(chunk_size[0], z - core_start[0])
        core_x = min(chunk_size[1], x - core_start[1])
        core_y = min(chunk_size[2], y - core_start[2])
        core_slice = tuple(
            [slice(margin, margin + s) for s in [core_z, core_x, core_y]]
        )
        chunk_slice = (
            slice(core_start[0], core_start[0] + core_z),
            slice(core_start[1], core_start[1] + core_x),
            slice(core_start[2], core_start[2] + core_y),
        )
        chunk_buf.fill(fill_value)

        slice_start = core_start - margin
        slice_start_p = np.maximum(0, slice_start)
        slice_offset = slice_start_p - slice_start
        valid_slice = layers_gpu[
            slice_start_p[0] : slice_start[0] + core_z + 2 * margin,
            slice_start_p[1] : slice_start[1] + core_x + 2 * margin,
            slice_start_p[2] : slice_start[2] + core_y + 2 * margin,
        ]
        vs_z, vs_x, vs_y = valid_slice.shape
        chunk_buf[
            slice_offset[0] : slice_offset[0] + vs_z,
            slice_offset[1] : slice_offset[1] + vs_x,
            slice_offset[2] : slice_offset[2] + vs_y,
        ] = valid_slice
        buffer_data_slice = (
            slice(core_z + 2 * margin),
            slice(core_x + 2 * margin),
            slice(core_y + 2 * margin),
        )
        # Also put margins in, defaulting to fill
        yield chunk_buf[buffer_data_slice], chunk_slice, core_slice


# %%
@profile
def main():
    s = (256 - 16, 512 - 16, 512 - 16)
    m = 16

    output = np.empty_like(layers)
    g = chunk_generator(layers, chunk_size=s, margin=m)
    for chunk, chunk_slice, core_slice in tqdm.tqdm(
        g, total=n_chunks(layers, chunk_size=s)
    ):
        if chunk.max() == 0:
            output[chunk_slice] = 0
            continue
        chunk = cp.array(chunk, dtype=cp.int16)
        skin = sla.mask.skin(chunk, (2, 2, 2))
        chunk[skin] += sla.int_noise(chunk[skin].shape, -135, 0)
        chunk = cp.clip(chunk, 0, 255)
        output[chunk_slice] = chunk[core_slice].get()


main()
