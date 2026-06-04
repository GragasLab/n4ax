"""Basic correctness: import, shapes, finiteness, 2D/3D, near-identity on no bias."""

import numpy as np

import n4ax


def test_import():
    assert callable(n4ax.n4)
    assert callable(n4ax.otsu_mask)


def test_3d_shape_and_finite(phantom):
    obs, _, _ = phantom
    corr = np.asarray(n4ax.n4(obs))
    assert corr.shape == obs.shape
    assert np.isfinite(corr).all()
    assert (corr >= 0).all()


def test_2d_runs():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.5, 1.5, size=(64, 64)).astype(np.float32)
    img[:5] = 0.0  # background
    corr = np.asarray(n4ax.n4(img))
    assert corr.shape == img.shape
    assert np.isfinite(corr).all()


def test_return_bias(phantom):
    obs, _, mask = phantom
    corr, bias = n4ax.n4(obs, mask=mask.astype(np.float32), return_bias=True)
    corr, bias = np.asarray(corr), np.asarray(bias)
    m = mask
    # corrected == image / exp(bias) inside the mask
    recon = obs / np.exp(bias)
    assert np.abs(corr[m] - recon[m]).max() < 1e-3


def test_recovers_bias(phantom):
    """N4 should flatten a known smooth bias: corrected tissue is more uniform
    (lower coefficient of variation) than the observed biased image."""
    obs, bias, mask = phantom
    corr = np.asarray(n4ax.n4(obs, mask=mask.astype(np.float32)))
    m = mask
    cv_before = obs[m].std() / obs[m].mean()
    cv_after = corr[m].std() / corr[m].mean()
    assert cv_after < cv_before


def test_otsu_mask(phantom):
    obs, _, mask = phantom
    om = np.asarray(n4ax.otsu_mask(obs)) > 0.5
    # Otsu foreground should agree with the true brain mask on most voxels
    agree = (om == mask).mean()
    assert agree > 0.95
