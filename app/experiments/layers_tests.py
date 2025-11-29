# %%
import sla_process as sla
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import cupy as cp
import cupyx.scipy.ndimage as cndi
import sla_process.util.kernels as kn

import napari

a = np.zeros((20, 100, 150), dtype=np.int16)
b = np.zeros((32, 50, 150), dtype=np.int16)

sla.collate_layers([a] * 3 + [b])
