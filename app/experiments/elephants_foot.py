# %%
import sla_process as sla
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import time
import cupy as cp
import tqdm

importlib.reload(sla)

import gc

import sys

# Dev
from line_profiler import profile
import matplotlib.pyplot as plt

# %%

DEVICE_TEMPLATE_PATH = Path(r"data\templates\PSM8KS.prz")
DEVICE_SUFFIX = DEVICE_TEMPLATE_PATH.suffix

layers = None


def main():
    global layers
    sample_path = Path(sys.argv[1])
    print(f"Loading {sample_path}...")
    sample_file = sla.SlicerFile(sample_path)
    print("Loading template...")
    output_file = sla.SlicerFile(DEVICE_TEMPLATE_PATH)
    output_path = sample_path.with_suffix(".8ks" + DEVICE_SUFFIX)

    print("Divvying out layers...")
    original_shape = sample_file.layers.shape
    layers = sample_file.layers
    layers_gpu = cp.asarray(sample_file.layers, dtype=cp.uint8)

    bottom_layers = layers[: sample_file.bottom_layer_count]
    reg_layers = layers[sample_file.bottom_layer_count :]
    bottom_layers_gpu = layers_gpu[: sample_file.bottom_layer_count]
    reg_layers_gpu = layers_gpu[sample_file.bottom_layer_count :]
    del sample_file

    chunk_size = (256, 512, 1024)
    gen = sla.chunks.chunk_generator(layers_gpu, chunk_size)

    for chunk, chunk_slice, core_slice in tqdm.tqdm(
        gen, total=sla.chunks.n_chunks(layers, chunk_size=chunk_size)
    ):
        if chunk.max() == 0:
            layers[chunk_slice] = 0
            continue
        skin = sla.mask.skin(chunk, (2, 2, 2))
        chunk[skin] += sla.int_noise(chunk[skin].shape, -135, 0)

        chunk = cp.clip(chunk, 0, 255)
        layers[chunk_slice] = chunk[core_slice].get()  # type: ignore

    print("Applying elephant's foot correction...")
    bottom_layers[:] = sla.elephants_foot(bottom_layers_gpu, 4).get()

    print("Updating thumbnail...")
    heatmap = sla.minimize_thumbnail(sla.layer_heatmap(layers))
    output_file.set_thumbnails(heatmap)

    # print("Restoring original layer size")
    # layers = sla.maximize_layers(layers, original_shape)

    print(f"Saving to {output_path}...")
    output_file.layers = layers
    output_file.save(output_path)


if __name__ == "__main__":
    main()
