# n4ax performance defect: `jnp.histogram` inside `otsu_mask` costs 440–1720 s

**Status: FIXED in `core.py` (validated). Root cause still NOT identified.**
**Filed:** 2026-08-16, from the triax qmap corpus rebuild.

## Summary

`n4()` spent **99% of its wall-clock in `otsu_mask()`**, not in the N4 algorithm.
On a real NKI T1w volume the N4 core runs in **0.16 s**; the mask cost **444 s**.

**The fix** (applied): replace `jnp.histogram` with a collision-free count — sort the
bin indices and difference the boundary positions — and jit the function.
**T1w: 444.6 s → 0.39 s in the real pipeline (1126×), mask bit-identical.**

The fix is justified by **output equivalence**, not by understanding the defect:
masks are bit-identical at every size tested, and the corrected volume differs by
less than `n4`'s own run-to-run nondeterminism (see below). The mechanism remains
open and is the reason this file still exists.

## THREE WRONG MECHANISMS (recorded so nobody re-derives them)

1. **"Zero background causes it."** Wrong — it is a trigger, not a cause. `data + 1e-3`
   changes nothing (444 s).
2. **"Colliding scatter-add on the zero bin."** Wrong — measured directly:
   11.5 M updates into 200 bins costs **0.08 ms** at maximum collision (all-same index)
   vs **0.02 ms** spread. Only 3.4×, and both are sub-millisecond.
3. **"Scatter-add is slow in general."** Wrong — see above, 0.08 ms.

**So the bottleneck is neither `searchsorted` (my replacement calls it on identical
shapes and is fast) nor the scatter (0.08 ms). Something else inside `jnp.histogram`
costs 440–1720 s and has not been identified.**

Untested hypothesis worth trying first: `jnp.histogram` builds
`weights = ones_like(a)` and scatters an **11.5 M-element weights array**, whereas the
discriminator scattered a scalar `1.0`. Array-valued vs scalar-broadcast scatter may
lower to very different kernels.

## Measurements (H100, jax 0.11.0, `CudaDevice(id=0)`)

Subject `PT027_NKI_sub-0504_ses-03`, three anat volumes:

| volume | shape | voxels | `n4()` total | `otsu_mask` | `n4(mask=)` core |
|---|---|---:|---:|---:|---:|
| FLAIR | (192,256,44) | 2.2 M | 49.1 s | **36.7 s** | 0.03 s |
| T1w | (176,256,256) | 11.5 M | 447.5 s | **444.5 s** | 0.16 s |
| T2w | (176,256,256) | 11.5 M | 451.5 s | **449.1 s** | 0.15 s |

≈ 930 s per subject, all of it mask.

### Content dependence

Same shape (176,256,256), same op sequence, same 40 iterations:

| input | `n4()` |
|---|---|
| synthetic uniform noise `abs(N(100,20))` | **4.9 s** |
| real T1w | **455 s** |
| synthetic noise × zeroed border | **476 s** |

So it is triggered by having a background, not by anatomy and not by size.

### Ruled out (each measured, not assumed)

- **Iteration count** — instrumented `core()` returns per-level counts: both
  synthetic and real run the full `[8,12,12,8] = 40`. No early exit either way.
- **JIT / shape recompilation** — compile is 9.6 s; recompile on a new shape 8.6 s.
- **CPU fallback** — `jax.devices() == [CudaDevice(id=0)]`.
- **I/O and archive extraction** — NIfTI read 0.24 s, session unzip 0.14 s.
- **Exact zeros / denormals at input** — `n4(real + 1e-3)` = 444 s (no change).
- **Voxel count alone** — bbox crop to 0.54 of voxels gives 2.7×, i.e. linear.

### Candidate fixes tried

| candidate | FLAIR (2.2 M) | speedup | T1w (11.5 M) | speedup | mask bit-identical |
|---|---:|---:|---:|---:|---|
| baseline `otsu_mask` | 96.3 s\* | — | 444.5 s | — | — |
| `jax.jit(otsu_mask)` | 37.2 s | 2.6× | **444.0 s** | **1.0×** | yes |
| jitted scatter histogram (no dynamic edges) | 37.0 s | 2.6× | (not reached) | — | yes |

\* the same baseline measured 36.7 s in another job. Comparing 37.2 s against the
*warm* 36.7 s baseline gives 1.0×, so the apparent FLAIR "2.6×" was a cold-vs-warm
artifact, consistent with T1w.

**BOTH CANDIDATES FAIL. Jitting buys nothing.** Masks are bit-identical, so they are
safe — they are simply not fixes.

### The strongest clue: superlinear scaling

| voxels | time |
|---:|---:|
| 2.2 M | ~37 s |
| 11.5 M | ~444 s |

5.2× the voxels costs **12×** the time. For a 200-bin histogram plus a threshold
compare — all linear-time ops — cost should scale linearly. Effective throughput on
T1w is ~0.1 MB/s for a 46 MB array, i.e. roughly four orders of magnitude below what
the arithmetic requires.

A jitted scatter histogram (no sort, no dynamic edges) measured *identical* to
`jnp.histogram` on FLAIR, which argues the histogram is **not** the hot spot at all.
Every op in the function is a linear reduction or elementwise compare. Something is
doing work the maths does not call for — that is what the profiler needs to find.

## The function

```python
def otsu_mask(volume, nbins: int = 200):
    v = jnp.asarray(volume, REAL)
    vmin, vmax = jnp.min(v), jnp.max(v)
    edges = jnp.linspace(vmin, vmax, nbins + 1)
    hist, _ = jnp.histogram(v, bins=edges)      # dynamic (traced) edges
    ...
    return (v > thr).astype(REAL)
```

Unjitted, and `jnp.histogram` with traced edges cannot use a fast path.

## Open questions for whoever picks this up

1. **Where does the time actually go?** Nobody has profiled it. Run
   `jax.profiler.trace` on `otsu_mask(real)` vs `otsu_mask(synthetic)` at identical
   shape and diff the op timeline. Everything above is wall-clock inference; this is
   the first step that would produce a mechanism rather than another hypothesis.
2. Why is it content-dependent *at all*? With fixed shapes, XLA op cost should not
   depend on values. fp32 denormals and NaN/Inf run at full rate on NVIDIA, so the
   usual suspects do not apply.
3. Why is the baseline variance so large (36.7 s vs 96.3 s for the same call)?
   Partly cold-vs-warm, but confirm.
4. Does `jnp.histogram` with traced edges fall back to a sort (O(n log n)) path?
   If so, a fixed-width scatter histogram should beat it — but it measured the same,
   which argues the histogram is *not* the hot spot.
5. **What explains superlinear scaling and ~0.1 MB/s effective throughput?** Every op
   here is a linear reduction or elementwise compare. Candidates worth checking first:
   a host round-trip per call (is `volume` staying a numpy array and being
   re-transferred, or worse, iterated?); memory pressure / thrashing against whatever
   else holds GPU memory in the caller's process; or an XLA fallback kernel. Compare
   `otsu_mask(np_array)` vs `otsu_mask(jnp.asarray(...))` and watch actual PCIe traffic
   — that one-line experiment may settle it.

## The fix, measured

| volume | voxels | zero-frac | before | after | speedup | mask identical |
|---|---:|---:|---:|---:|---:|---|
| FLAIR | 2.2 M | 0.058 | 71.8 s | 0.005 s | 15099× | yes |
| T1w | 11.5 M | 0.619 | 440.6 s | 0.031 s | 14048× | yes |
| T1w (other run) | 11.5 M | 0.619 | 1721.7 s | 0.040 s | 43074× | yes |
| **T1w, real pipeline** | 11.5 M | 0.619 | **444.6 s** | **0.39 s** | **1126×** | **yes** |

Baseline variance is enormous (440.6 / 1721.7 s for the same call) — another
unexplained property of the original path. The fixed path is stable.

Two collision-free formulations were tried; both bit-identical, both fine:
sort+boundaries (0.031 s) and per-bin reduction (0.022 s) at 11.5 M. `core.py` uses
sort+boundaries.

## `n4` is nondeterministic (pre-existing, unrelated)

Calling `n4()` **twice with identical mask and identical input** gives
`max rel = 6.45e-05`, not bit-identical — `_bspline_fit` accumulates via float atomics
whose order varies. The patched-vs-original difference (3.2–4.2e-05) is *smaller* than
this, which is how the patch was cleared.

Consequence beyond this bug: **qmap targets are not bit-reproducible run to run.**
Same config and same data agree to ~1e-4 relative, not exactly. Relevant to any
reproducibility claim about the corpus.

## Reproduction

```bash
# on Jean-Zay
sbatch $SCRATCH/bench_otsu.slurm        # jit variants + bit-identity check
sbatch $SCRATCH/verify_mask.slurm       # otsu vs core split, all 3 volumes
```
Scripts also in `$CLAUDE_JOB_DIR/tmp/bench_{n4,real,disc,fix,iters,otsu}.slurm`.

## Why it matters beyond triax

n4ax is intended for open-source release. A bias-field correction package that takes
7 minutes per volume on real MRI — while advertising milliseconds — is not
releasable. The defect only shows on real data, which is exactly the case every
external user will hit first.

For the corpus rebuild specifically it is ~300 GPU-h; that is being worked around by
caching prep output, not by fixing this.
