# %%
import sla_process as sla
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import time

importlib.reload(sla)

import sys

# Dev
from line_profiler import profile
import matplotlib.pyplot as plt

# %%
times = []
times.append(time.time())

DEVICE_TEMPLATE_PATH = Path(r"data\templates\PSM8KS.prz")
DEVICE_SUFFIX = DEVICE_TEMPLATE_PATH.suffix

layers = None

default_path = Path(r"c:\Users\vinitlee\Desktop\bottom_draft_test.sl1")


@profile
def main():
    global layers
    sample_path = Path(sys.argv[1])
    # sample_path = default_path
    print(f"Loading {sample_path}...")
    sample_file = sla.SlicerFile(sample_path)
    print("Loading template...")
    output_file = sla.SlicerFile(DEVICE_TEMPLATE_PATH)
    output_path = sample_path.with_suffix(".mod" + DEVICE_SUFFIX)

    print("Divvying out layers...")
    times.append(time.time())
    original_shape = sample_file.layers.shape
    layers = sla.minimize_layers(sample_file.layers)

    bottom_layers = layers[: sample_file.bottom_layer_count]
    reg_layers = layers[sample_file.bottom_layer_count :]
    del sample_file
    times.append(time.time())
    print(f"Took {times[-1] - times[-2]}s to execute")

    print("Computing masks")
    model = sla.mask.model(layers)
    times.append(time.time())
    skin = sla.mask.skin(layers, thickness=(1, 1, 1), approach="cp_kernel")
    times.append(time.time())
    print(f"Took {times[-1] - times[-2]}s to execute")
    reg_skin = skin[len(bottom_layers) :]

    print("Applying noise")
    reg_layers[reg_skin] += sla.int_noise(reg_layers[reg_skin].shape, -135, 0)

    print("Applying elephant's foot correction...")
    # bottom_layers[:] = sla.beveled_elephants_foot(
    #     bottom_layers, 4, sample_file.bottom_layer_count
    # )
    bottom_layers[:] = sla.elephants_foot(bottom_layers, 4)

    print("Updating thumbnail...")
    heatmap = sla.minimize_thumbnail(sla.layer_heatmap(layers))
    output_file.set_thumbnails(heatmap)

    print("Restoring original layer size")
    layers = sla.maximize_layers(layers, original_shape)

    print(f"Saving to {output_path}...")
    output_file.layers = layers
    output_file.save(output_path)

    times.append(time.time())
    print(f"Total time: {times[-1] - times[0]}")


if __name__ == "__main__":
    main()
