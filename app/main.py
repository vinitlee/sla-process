# %%
import sla_process as sla
import matplotlib.pyplot as plt
from pathlib import Path
import importlib
import numpy as np
import scipy.ndimage as ndi
import cupy as cp
import cupyx.scipy.ndimage as cndi
import sla_process.util.kernels as kn
from line_profiler import profile
import time

importlib.reload(sla)

# %%
sample_path = Path(r"example_data\sl1\sample.sl1")
sample_file = sla.SlicerFile(sample_path)
output_path = sample_path.with_suffix(".mod" + sample_path.suffix)

# %%

layers = sample_file.layers[:]
bottom_layers = layers[: sample_file.bottom_layer_count]
top_layers = layers[sample_file.bottom_layer_count :]
# %%
# model = sla.mask.model(layers)
t0 = time.time()
skin = sla.mask.skin(layers, approach="cp_kernel", thickness=(2, 2, 2))
print(time.time() - t0)
plt.imshow(skin[30, 170:300, 170:300])
plt.show()

# skin = sla.mask.skin(layers, thickness=(2, 4, 1))
# plt.imshow(skin[30, 170:300, 170:300])
# plt.show()

# top_skin = skin[sample_file.bottom_layer_count :]
# top_layers[top_skin] = sla.noise(top_layers[top_skin].shape, 200, 255)


# %%
def main():
    pass


if __name__ == "__main__":
    main()
