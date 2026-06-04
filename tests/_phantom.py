"""Shared phantom: continuous brain-like tissue inside an ellipsoid times a smooth
multiplicative bias field — the realistic case N4 is designed for."""

import numpy as np


def make_phantom(shape=(24, 64, 64), seed=0):
    s, h, w = shape
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:s, 0:h, 0:w].astype(np.float32)
    cz, cy, cx = s / 2, h / 2, w / 2
    ell = ((z - cz) / (s * 0.45)) ** 2 + ((y - cy) / (h * 0.42)) ** 2 + ((x - cx) / (w * 0.42)) ** 2
    mask = ell <= 1.0
    tissue = 1.4 + 0.5 * np.sin(3 * np.pi * y / h) * np.cos(3 * np.pi * x / w) + 0.3 * np.cos(2 * np.pi * z / s)
    tissue = np.clip(tissue + rng.standard_normal(shape).astype(np.float32) * 0.04, 0.3, None)
    bias = 1.0 + 0.30 * np.sin(2 * np.pi * y / h) + 0.20 * np.cos(2 * np.pi * x / w) + 0.15 * (z / s)
    obs = tissue * bias
    obs[~mask] = 0.0
    return obs.astype(np.float32), bias.astype(np.float32), mask
