"""N4 bias field correction in pure JAX — a fast, GPU-friendly drop-in match for
ITK / SimpleITK's ``N4BiasFieldCorrectionImageFilter``.

Algorithm (Tustison 2010 N4 = N3 histogram sharpening + multi-resolution B-spline):

    u = log(v) over the mask;  B = 0  (log bias field)
    for each fitting level (mesh = 1, 2, 4, 8):
        repeat:
            uc = u - B
            E  = sharpen(uc)               # N3 histogram deconvolution (Wiener, FFT)
            S  = bspline_fit(uc - E)        # cubic B-spline least-squares (Lee MBA)
            B  = B + over_relax * S
    corrected = exp(u - B)                  # == v / exp(B)

Every building block matches ITK's implementation (parametric coords, cubic
B-spline weights, Lee-MBA accumulation, N3 Wiener deconvolution). Two ideas make
it fast on a GPU without any custom kernel:

* the B-spline fit is **separable** (weights depend only on the per-axis index and
  the Lee denominator factorises), so the per-voxel scatter into the control
  lattice becomes three small dense matmuls per axis — no atomic contention;
* the sharpening histogram is **privatised** over K lanes so the value-scatter
  doesn't serialise on atomics.

``over_relax > 1`` accelerates N4's (slow, monotone) crawl to the *same* fixed
point (S = 0 there, so ``B += a*S`` is fixed-point invariant), reaching ITK's
result in far fewer iterations.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


SPLINE_ORDER = 3
REAL = jnp.float32  # the iteration is float-precision-insensitive (verified vs float64)


# ----------------------------- Otsu mask ------------------------------------
@functools.partial(jax.jit, static_argnums=(1,))
def otsu_mask(volume, nbins: int = 200):
    """Binary foreground mask via Otsu's threshold (matches ITK ``OtsuThreshold``
    with insideValue=0/outsideValue=1: foreground = intensity above threshold).

    The bin counts are obtained by sorting the bin indices and differencing the
    boundary positions, NOT by ``jnp.histogram``. ``jnp.histogram`` ends in
    ``zeros(nbins).at[bin_idx].add(weights)`` — a scatter-add. On any image with a
    dominant background (i.e. every real MRI, where the air is exactly 0) a large
    fraction of the updates target one bin, and the colliding atomics serialise:
    measured 71.8 s vs 0.005 s here for 2.2 M voxels, with bit-identical output.
    Cost also grew superlinearly with voxel count, and jitting alone did not help.
    Keep this collision-free; do not "simplify" it back to ``jnp.histogram``.
    """
    v = jnp.asarray(volume, REAL)
    vmin, vmax = jnp.min(v), jnp.max(v)
    edges = jnp.linspace(vmin, vmax, nbins + 1)
    idx = jnp.clip(jnp.searchsorted(edges, v.reshape(-1), side="right") - 1, 0, nbins - 1)
    boundaries = jnp.searchsorted(jnp.sort(idx), jnp.arange(nbins + 1, dtype=idx.dtype), side="left")
    hist = jnp.diff(boundaries).astype(REAL)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = jnp.cumsum(hist)
    wb = w
    wf = w[-1] - w
    csum = jnp.cumsum(hist * centers)
    mb = jnp.where(wb > 0, csum / jnp.where(wb > 0, wb, 1), 0.0)
    mf = jnp.where(wf > 0, (csum[-1] - csum) / jnp.where(wf > 0, wf, 1), 0.0)
    between = wb * wf * (mb - mf) ** 2
    thr = centers[jnp.argmax(between)]
    return (v > thr).astype(REAL)


# ----------------------------- N3 sharpening --------------------------------
@functools.partial(jax.jit, static_argnums=(2,))
def _sharpen(uc, mask, nbins=200, fwhm=0.15, wiener=0.01):
    """N3 histogram-deconvolution sharpening of the masked log image ``uc``."""
    m = mask > 0.5
    vmin = jnp.min(jnp.where(m, uc, jnp.inf))
    vmax = jnp.max(jnp.where(m, uc, -jnp.inf))
    slope = (vmax - vmin) / (nbins - 1)

    # parzen (2-bin linear) histogram, privatised over K lanes to avoid atomic
    # contention (the value-scatter was ~100% of sharpen's cost otherwise).
    cidx = (uc - vmin) / slope
    idx = jnp.floor(cidx).astype(jnp.int32)
    off = cidx - idx
    w = m.astype(REAL).reshape(-1)
    idxf = idx.reshape(-1)
    offf = off.reshape(-1)
    K = 256
    lane = jnp.arange(idxf.shape[0], dtype=jnp.int32) % K
    Hp = jnp.zeros((K, nbins), REAL)
    Hp = Hp.at[lane, jnp.clip(idxf, 0, nbins - 1)].add(w * (1.0 - offf))
    Hp = Hp.at[lane, jnp.clip(idxf + 1, 0, nbins - 1)].add(w * offf)
    H = jnp.sum(Hp, axis=0)

    # zero-padded FFT (npad >= 2*nbins) so the Gaussian deconvolution doesn't
    # suffer circular wraparound (that wraparound otherwise breaks convergence).
    npad = 1
    while npad < 2 * nbins:
        npad *= 2
    Hp_ = jnp.zeros((npad,), REAL).at[:nbins].set(H)
    k = jnp.arange(npad).astype(REAL)
    ln2 = jnp.log(2.0)
    scaled_fwhm = fwhm / slope
    exp_factor = 4.0 * ln2 / (scaled_fwhm**2)
    scale_factor = 2.0 * jnp.sqrt(ln2 / jnp.pi) / scaled_fwhm
    d = jnp.where(k > npad / 2, k - npad, k)
    F = scale_factor * jnp.exp(-(d**2) * exp_factor)

    Ff = jnp.fft.fft(F)
    Gf = jnp.conj(Ff) / (jnp.abs(Ff) ** 2 + wiener)  # Wiener filter
    Uhat = jnp.clip(jnp.real(jnp.fft.ifft(jnp.fft.fft(Hp_) * Gf)), 0.0, None)

    centers = vmin + k * slope
    num = jnp.real(jnp.fft.ifft(jnp.fft.fft(Uhat * centers) * Ff))
    den = jnp.real(jnp.fft.ifft(jnp.fft.fft(Uhat) * Ff))
    E = (num / jnp.where(jnp.abs(den) > 1e-10, den, 1e-10))[:nbins]

    ci = jnp.clip(cidx, 0.0, nbins - 1.0)
    lo = jnp.floor(ci).astype(jnp.int32)
    fr = ci - lo
    hi = jnp.clip(lo + 1, 0, nbins - 1)
    return jnp.where(m, E[lo] * (1.0 - fr) + E[hi] * fr, 0.0)


# ----------------------------- B-spline fit ---------------------------------
def _bspline_w(frac):
    """Order-3 uniform B-spline weights for the 4 controls span..span+3."""
    f = frac
    return jnp.stack(
        [
            (1.0 - f) ** 3 / 6.0,
            (3.0 * f**3 - 6.0 * f**2 + 4.0) / 6.0,
            (-3.0 * f**3 + 3.0 * f**2 + 3.0 * f + 1.0) / 6.0,
            f**3 / 6.0,
        ],
        axis=-1,
    )


def _axis_mats(n, ncp, mesh):
    """Sparse per-axis weight matrices (ncp x n) at powers 1/2/3 of the cubic
    weights. The 3D Lee fit is separable -> these turn the scatter into matmuls."""
    i = jnp.arange(n).astype(REAL)
    p = jnp.clip(i / max(n - 1, 1) * mesh, 0.0, float(mesh))  # max(): handle singleton axes (2D)
    span = jnp.clip(jnp.floor(p).astype(jnp.int32), 0, mesh - 1)
    w = _bspline_w(p - span)
    rows = (span[:, None] + jnp.arange(4)[None, :]).reshape(-1)
    cols = jnp.repeat(jnp.arange(n), 4)

    def mk(power):
        return jnp.zeros((ncp, n), REAL).at[rows, cols].add((w**power).reshape(-1))

    return mk(1), mk(2), mk(3)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _bspline_fit(r, mask, ncp_shape, mesh):
    """Cubic B-spline Lee-MBA fit of residual ``r`` over the mask, evaluated densely.
    Separable formulation: identical math to the per-voxel scatter, as matmuls."""
    s, h, w = r.shape
    ncpz, ncpy, ncpx = ncp_shape
    Wz1, Wz2, Wz3 = _axis_mats(s, ncpz, mesh)
    Wy1, Wy2, Wy3 = _axis_mats(h, ncpy, mesh)
    Wx1, Wx2, Wx3 = _axis_mats(w, ncpx, mesh)

    mvox = (mask > 0.5).astype(REAL)
    sz, sy, sx = jnp.sum(Wz2, 0), jnp.sum(Wy2, 0), jnp.sum(Wx2, 0)  # Lee denom factorises
    g = (r * mvox) / (sz[:, None, None] * sy[None, :, None] * sx[None, None, :])

    num = jnp.einsum("cx,abx->abc", Wx3, jnp.einsum("by,ayx->abx", Wy3, jnp.einsum("az,zyx->ayx", Wz3, g)))
    den = jnp.einsum("cx,abx->abc", Wx2, jnp.einsum("by,ayx->abx", Wy2, jnp.einsum("az,zyx->ayx", Wz2, mvox)))
    phi = num / jnp.where(den > 1e-12, den, 1e-12)

    p1 = jnp.einsum("az,abc->zbc", Wz1, phi)
    p2 = jnp.einsum("by,zbc->zyc", Wy1, p1)
    return jnp.einsum("cx,zyc->zyx", Wx1, p2)


def _conv_cv(b_prev, b, maskb, cnt):
    """ITK convergence measure: CV = std/mean of exp(B_prev - B_curr) over the mask."""
    r = jnp.exp(jnp.where(maskb, b_prev - b, 0.0))
    mu = jnp.sum(jnp.where(maskb, r, 0.0)) / cnt
    var = jnp.sum(jnp.where(maskb, (r - mu) ** 2, 0.0)) / (cnt - 1.0)
    return jnp.sqrt(var) / mu


# ----------------------------- driver ---------------------------------------
@functools.cache
def _compiled(iters, nbins):
    @jax.jit
    def core(v, mask, fwhm, wiener, threshold, tiny, over_relax):
        maskb = mask > 0.5
        cnt = jnp.sum(maskb.astype(REAL))
        u = jnp.log(jnp.clip(v, tiny, None))
        b = jnp.zeros_like(u)
        used = []
        for lvl, nit in enumerate(iters):
            mesh = 2**lvl
            ncp = (mesh + SPLINE_ORDER,) * 3

            def cond(c, _nit=nit):
                _, i, cv = c
                return (i < _nit) & (cv > threshold)

            def body(c, _ncp=ncp, _mesh=mesh):
                bp, i, _ = c
                uc = u - bp
                e = _sharpen(uc, mask, nbins, fwhm, wiener)
                s = _bspline_fit(jnp.where(maskb, uc - e, 0.0), mask, _ncp, _mesh)
                bn = bp + over_relax * s
                return (bn, i + 1, _conv_cv(bp, bn, maskb, cnt))

            b, i, _ = jax.lax.while_loop(cond, body, (b, 0, jnp.array(jnp.inf, REAL)))
            used.append(i)
        corrected = jnp.where(maskb, jnp.exp(u - b), v)
        return corrected, b, jnp.stack(used)

    return core


def n4(
    image,
    mask=None,
    *,
    iters=(8, 12, 12, 8),
    over_relax: float = 1.8,
    conv_threshold: float = 0.0,
    nbins: int = 200,
    fwhm: float = 0.15,
    wiener: float = 0.01,
    tiny: float = 1e-6,
    return_bias: bool = False,
):
    """N4 bias field correction.

    Args:
        image: ND array (2D or 3D) of intensities (>= 0).
        mask: optional binary foreground mask; if ``None``, an Otsu mask is used.
        iters: max iterations per fitting level (mesh 1, 2, 4, 8). ``conv_threshold``
            (ITK-style CV) can stop a level early; the default uses fixed counts
            with over-relaxation for speed.
        over_relax: relaxation factor (>= 1) accelerating the crawl to the fixed point.
        nbins/fwhm/wiener: N3 sharpening parameters (ITK defaults).

    Returns:
        corrected image (== image / exp(bias)); also the log bias field if ``return_bias``.
    """
    image = jnp.asarray(image, REAL)
    in3d = image.ndim == 3
    v = image if in3d else image[None]
    mask = otsu_mask(v) if mask is None else jnp.asarray(mask, REAL).reshape(v.shape)
    core = _compiled(tuple(iters), nbins)
    corrected, bias, _ = core(
        v,
        mask,
        jnp.asarray(fwhm, REAL),
        jnp.asarray(wiener, REAL),
        jnp.asarray(conv_threshold, REAL),
        jnp.asarray(tiny, REAL),
        jnp.asarray(over_relax, REAL),
    )
    if not in3d:
        corrected, bias = corrected[0], bias[0]
    return (corrected, bias) if return_bias else corrected
