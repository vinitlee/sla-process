import scipy.ndimage as ndi
import cupyx.scipy.ndimage as cndi
import numpy as np
import cupy as cp
import sla_process.core.masking as mask
import sla_process.util.math as sla_math


def noise(shape, a_min=-1, a_max=1) -> cp.typing.NDArray:
    gen = cp.random.random(shape)
    gen *= a_max - a_min
    gen += a_min
    return gen


def int_noise(shape, a_min=-128, a_max=128) -> cp.typing.NDArray:
    gen = cp.random.random(shape)
    gen *= a_max - a_min
    gen += a_min
    gen = gen.astype(np.int16)
    return gen


def erosion_noise(layers: np.typing.NDArray, depth: int, period: float):
    temp_layers = layers.copy()

    model_mask = mask.model(layers)
    skin_mask = mask.skin(layers, 1)
    band_mask = mask.skin(layers, depth)

    noise_src = noise(temp_layers[skin_mask].shape, 0, 1)
    noise_src = np.where(noise_src > 1 / period, 255, 0)
    temp_layers[skin_mask] = noise_src


def weighted_additive_noise(
    layers: np.typing.NDArray, weights: np.typing.NDArray, noise: np.typing.NDArray
):
    if not (layers.shape == weights.shape == noise.shape):
        raise Exception("All arrays must be of the same shape.")


def displace_with_noise(layers: cp.typing.NDArray) -> cp.typing.NDArray:
    # ndi.map_coordinates()
    pass


def noisy_greys(
    layers: cp.typing.NDArray, amp: int = 20, min_th: int = 0, max_th: int = 250
):
    """
    min_th and max_th are non-inclusive
    """

    mask = (layers > min_th) & (layers < max_th)
    layers[mask] = sla_math.pingpong(
        layers[mask] + int_noise(layers[mask].shape, -amp // 2, amp // 2),
        min_th + 1,
        max_th - 1,
    )

    return layers
