"""Benchmark + illustrations on real NKI T1w volumes (FOMO300K).

Runs ITK N4 (CPU) and n4ax (current JAX backend) on several raw NKI T1w scans
with the SAME Otsu mask, reports per-volume time + n4ax-vs-ITK agreement, and
saves figures (raw | n4ax | ITK | bias) per subject + a multi-subject grid.

Run on GPU:        python bench_nki.py
Time n4ax on CPU:  JAX_PLATFORMS=cpu python bench_nki.py --skip-itk --skip-fig --tag cpu
"""

import argparse
import os
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np


FOMO = "/lustre/fsn1/projects/rech/ijy/upd68za/FOMO300K/PT027_NKI"
OUT = Path(os.environ.get("N4AX_ASSETS", "/lustre/fswork/projects/rech/hlp/uha64uw/gragas/n4ax/assets"))


def load_t1w(subject):
    import nibabel as nib

    zpath = sorted(Path(f"{FOMO}/{subject}").glob("ses-*.zip"))[0]
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith("_T1w.nii.gz"))
        data = z.read(name)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(data)
        tmp = f.name
    vol = np.asanyarray(nib.load(tmp).dataobj).astype(np.float32)
    os.unlink(tmp)
    vol = np.clip(vol, 0, None)
    # SimpleITK array order is (z,y,x); nibabel is (x,y,z) -> move for consistency
    return np.ascontiguousarray(np.moveaxis(vol, -1, 0))


def itk_n4(vol, mask):
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(vol)
    mk = sitk.GetImageFromArray(mask.astype(np.uint8))
    c = sitk.N4BiasFieldCorrectionImageFilter()
    c.SetMaximumNumberOfIterations([50, 50, 30, 20])
    t = time.time()
    out = c.Execute(img, mk)
    return sitk.GetArrayFromImage(out).astype(np.float32), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subjects", nargs="+", default=["sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005", "sub-0006"]
    )
    ap.add_argument("--skip-itk", action="store_true")
    ap.add_argument("--skip-fig", action="store_true")
    ap.add_argument("--n-fig", type=int, default=4)
    ap.add_argument("--tag", default="gpu")
    args = ap.parse_args()

    import jax

    import n4ax

    backend = jax.default_backend()
    dev = str(jax.devices()[0])
    OUT.mkdir(parents=True, exist_ok=True)

    rows, fig_data = [], []
    for sub in args.subjects:
        vol = load_t1w(sub)
        mask = np.asarray(n4ax.otsu_mask(vol)) > 0.5

        # n4ax (current backend), warm then timed
        n4ax.n4(vol, mask=mask.astype(np.float32))  # warm/compile
        t = time.time()
        corr = np.asarray(n4ax.n4(vol, mask=mask.astype(np.float32)))
        jax.block_until_ready(corr)
        t_jax = time.time() - t
        _, bias = n4ax.n4(vol, mask=mask.astype(np.float32), return_bias=True)
        bias = np.asarray(bias)

        t_itk, rel = float("nan"), float("nan")
        itk = None
        if not args.skip_itk:
            itk, t_itk = itk_n4(vol, mask)
            ratio = corr[mask] / np.clip(itk[mask], 1e-6, None)
            rel = float(
                np.mean(np.abs(corr[mask] / np.median(ratio) - itk[mask]) / np.clip(np.abs(itk[mask]), 1e-6, None))
            )

        rows.append((sub, vol.shape, int(mask.sum()), t_itk, t_jax, rel))
        print(
            f"{sub} {vol.shape} mask={int(mask.sum())}: ITK={t_itk:.2f}s  n4ax({backend})={t_jax * 1000:.0f}ms  rel={rel * 100:.2f}%",
            flush=True,
        )
        if not args.skip_fig and len(fig_data) < args.n_fig:
            fig_data.append((sub, vol, corr, itk, bias, mask))

    print(f"\n=== summary ({backend}) on {dev} ===")
    its = [r[3] for r in rows if r[3] == r[3]]
    jts = [r[4] for r in rows]
    rels = [r[5] for r in rows if r[5] == r[5]]
    if its:
        print(f"ITK CPU   mean: {np.mean(its):.2f} s")
    print(f"n4ax {backend:4s} mean: {np.mean(jts) * 1000:.0f} ms")
    if its:
        print(f"speedup        : {np.mean(its) / np.mean(jts):.0f}x")
    if rels:
        print(f"n4ax vs ITK rel-err mean: {np.mean(rels) * 100:.2f}%")

    if fig_data:
        _figures(fig_data, args.tag)
    # persist the timing rows for the README table
    np.save(
        OUT / f"timings_{args.tag}.npy",
        np.array([(r[0], r[3], r[4], r[5]) for r in rows], dtype=object),
        allow_pickle=True,
    )


def _figures(fig_data, tag):
    from copy import copy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def black_cmap(name):  # masked / out-of-brain voxels render black, not white
        c = copy(plt.get_cmap(name))
        c.set_bad("black")
        return c

    gray, viri = black_cmap("gray"), black_cmap("viridis")

    def draw(ax, im2d, m2d, cmap, masked=True):
        d = np.ma.masked_where(~m2d, im2d) if masked else im2d
        vlo, vhi = (np.percentile(im2d[m2d], 1), np.percentile(im2d[m2d], 99)) if m2d.any() else (0, 1)
        ax.set_facecolor("black")
        ax.imshow(d, cmap=cmap, vmin=vlo, vmax=vhi)
        ax.axis("off")

    # per-subject panels: raw | n4ax | ITK | bias
    for sub, vol, corr, itk, bias, mask in fig_data:
        zc = vol.shape[0] // 2
        m = mask[zc]
        panels = [("raw T1w", vol[zc], gray, False), ("n4ax corrected", corr[zc], gray, True)]
        if itk is not None:
            panels.append(("ITK corrected", itk[zc], gray, True))
        panels.append(("n4ax bias field", np.exp(bias[zc]), viri, True))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), facecolor="black")
        for ax, (ti, im2d, cm, mk) in zip(axes, panels):
            draw(ax, im2d, m, cm, mk)
            ax.set_title(ti, fontsize=10, color="white")
        fig.suptitle(f"NKI {sub} — axial z={zc}", fontsize=11, color="white")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(OUT / f"nki_{sub}_{tag}.png", dpi=120, bbox_inches="tight", facecolor="black")
        plt.close(fig)

    # multi-subject grid: raw (top) vs n4ax-corrected (bottom)
    n = len(fig_data)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), facecolor="black")
    axes = np.atleast_2d(axes)
    for j, (sub, vol, corr, itk, bias, mask) in enumerate(fig_data):
        zc = vol.shape[0] // 2
        m = mask[zc]
        draw(axes[0, j], vol[zc], m, gray, masked=False)
        axes[0, j].set_title(sub, fontsize=9, color="white")
        draw(axes[1, j], corr[zc], m, gray, masked=True)
    fig.text(0.5, 0.965, "NKI raw (top) vs n4ax-corrected (bottom)", ha="center", fontsize=12, color="white")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"nki_grid_{tag}.png", dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
