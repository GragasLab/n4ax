"""Ground-truth test: n4ax must match SimpleITK's N4 (the reference implementation).

Uses the SAME mask for both (so we compare the N4 solve, not the masking), and
compares the corrected image with its global scale removed (N4's bias field is
defined up to a constant; downstream pipelines intensity-normalise anyway)."""

import numpy as np
import pytest

import n4ax


sitk = pytest.importorskip("SimpleITK")
from _phantom import make_phantom  # noqa: E402


def _itk_n4(obs, mask, iters=(50, 50, 30, 20)):
    img = sitk.GetImageFromArray(obs.astype(np.float32))
    mk = sitk.GetImageFromArray(mask.astype(np.uint8))
    c = sitk.N4BiasFieldCorrectionImageFilter()
    c.SetMaximumNumberOfIterations([int(i) for i in iters])
    out = c.Execute(img, mk)
    return sitk.GetArrayFromImage(out).astype(np.float64)


def _scaled_relerr(a, b, m):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ratio = a[m] / np.clip(b[m], 1e-6, None)
    rel = np.abs(a[m] / np.median(ratio) - b[m]) / np.clip(np.abs(b[m]), 1e-6, None)
    return rel


def test_matches_simpleitk():
    obs, _, mask = make_phantom()
    itk = _itk_n4(obs, mask)
    jax_corr = np.asarray(n4ax.n4(obs, mask=mask.astype(np.float32)))
    rel = _scaled_relerr(jax_corr, itk, mask)
    assert rel.mean() < 0.015, f"mean rel-err {rel.mean() * 100:.2f}% too high"
    assert np.percentile(rel, 95) < 0.05, f"p95 rel-err {np.percentile(rel, 95) * 100:.2f}% too high"


def test_closer_to_itk_than_uncorrected():
    """n4ax's correction must be much closer to ITK's than doing nothing."""
    obs, _, mask = make_phantom(seed=1)
    itk = _itk_n4(obs, mask)
    jax_corr = np.asarray(n4ax.n4(obs, mask=mask.astype(np.float32)))
    err_jax = _scaled_relerr(jax_corr, itk, mask).mean()
    err_none = _scaled_relerr(obs, itk, mask).mean()
    assert err_jax < 0.25 * err_none
