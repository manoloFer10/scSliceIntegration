"""Per-stack metric summary plot.

Reads the `cell_metrics_per_object.csv` produced by the pipeline (or computes
it on the fly if missing) and renders a multi-panel figure summarising every
metric group for that stack.

Usage:
    python visualize_stack.py "NIfTIs/Day_1/Alginate/well 1_Alginate_z stack 1"
    python visualize_stack.py <stack_dir> --output summary.png
    python visualize_stack.py <stack_dir> --recompute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from sc3d import config
from sc3d.core.pipeline import process_stack
from sc3d.io.discovery import walk_segmentations
from sc3d.io.nifti import NiftiStackLoader


SHAPE_HISTS = ("volume_phys", "sphericity", "elongation", "flatness",
               "equiv_spherical_radius", "surface_to_volume_ratio")
INTENSITY_HISTS = ("fluor_mean", "fluor_std", "fluor_sum", "fluor_cv",
                   "fluor_skewness", "fluor_kurtosis")


def _load_or_compute(stack_dir: Path, recompute: bool) -> pd.DataFrame:
    csv = stack_dir / config.PER_STACK_CSV
    if csv.exists() and not recompute:
        return pd.read_csv(csv)

    loader = NiftiStackLoader()
    refs = list(walk_segmentations(stack_dir))
    if not refs:
        raise FileNotFoundError(f"No *_segmentation.nii under {stack_dir}")
    if len(refs) > 1:
        raise ValueError(f"Expected one segmentation in {stack_dir}, got {len(refs)}")
    rows = process_stack(loader, refs[0])
    return pd.DataFrame(rows)


def _hist_panel(ax, values: pd.Series, title: str) -> None:
    values = values.dropna()
    if values.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(values, bins=min(40, max(8, int(np.sqrt(len(values))))),
            color="steelblue", edgecolor="white")
    ax.axvline(values.median(), color="crimson", linestyle="--", linewidth=1,
               label=f"median={values.median():.3g}")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)


def _scatter(ax, df: pd.DataFrame, x: str, y: str, *, log_x=False, log_y=False) -> None:
    sub = df[[x, y]].dropna()
    ax.scatter(sub[x], sub[y], s=8, alpha=0.4, color="darkslategray")
    if log_x: ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel(x, fontsize=8)
    ax.set_ylabel(y, fontsize=8)
    ax.tick_params(labelsize=8)


def _summary_text(df: pd.DataFrame) -> str:
    n = len(df)
    if n == 0:
        return "No objects."
    return (
        f"objects: {n}\n"
        f"voxels (median): {int(df['n_voxels'].median())}\n"
        f"volume (sum): {df['volume_phys'].sum():.4g}\n"
        f"sphericity (median): {df['sphericity'].median():.3f}\n"
        f"fluor_mean (median): {df['fluor_mean'].median():.2f}\n"
        f"centroid_disp (median): {df['centroid_displacement'].median():.4g}\n"
    )


def make_figure(df: pd.DataFrame, title: str) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 6)

    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Row 0: text summary + n_voxels distribution + label-vs-volume curve
    ax_text = fig.add_subplot(gs[0, :2])
    ax_text.axis("off")
    ax_text.text(0.0, 1.0, _summary_text(df), va="top", ha="left",
                 family="monospace", fontsize=9)

    ax_nv = fig.add_subplot(gs[0, 2:4])
    _hist_panel(ax_nv, np.log10(df["n_voxels"].clip(lower=1)), "log10(n_voxels)")

    ax_curve = fig.add_subplot(gs[0, 4:])
    sub = df.sort_values("label")
    ax_curve.plot(sub["label"], sub["volume_phys"], marker=".", linewidth=0.7,
                  markersize=3, color="darkblue")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("label (size rank, 1 = largest)", fontsize=8)
    ax_curve.set_ylabel("volume_phys", fontsize=8)
    ax_curve.set_title("size-rank curve")
    ax_curve.tick_params(labelsize=8)

    # Row 1: shape histograms
    for i, col in enumerate(SHAPE_HISTS):
        ax = fig.add_subplot(gs[1, i])
        _hist_panel(ax, df[col], col)

    # Row 2: intensity histograms
    for i, col in enumerate(INTENSITY_HISTS):
        ax = fig.add_subplot(gs[2, i])
        _hist_panel(ax, df[col], col)

    # Row 3: scatter relationships + derived
    ax = fig.add_subplot(gs[3, 0:2])
    _scatter(ax, df, "volume_phys", "fluor_sum", log_x=True, log_y=True)
    ax.set_title("volume vs fluor_sum")

    ax = fig.add_subplot(gs[3, 2:4])
    _scatter(ax, df, "sphericity", "roundness")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.7, linestyle=":")
    ax.set_title("sphericity vs roundness (QA)")

    ax = fig.add_subplot(gs[3, 4:])
    _scatter(ax, df, "fluor_mean", "centroid_displacement")
    ax.set_title("fluor_mean vs centroid_displacement")

    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stack_dir", type=Path, help="Folder containing *_segmentation.nii")
    ap.add_argument("--output", type=Path, default=None,
                    help="PNG path (default: <stack_dir>/metrics_summary.png)")
    ap.add_argument("--recompute", action="store_true",
                    help="Ignore existing per-stack CSV and recompute from NIfTI")
    ap.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = ap.parse_args(argv)

    if not args.stack_dir.is_dir():
        ap.error(f"{args.stack_dir} is not a directory")

    df = _load_or_compute(args.stack_dir, args.recompute)
    out = args.output or (args.stack_dir / "metrics_summary.png")
    fig = make_figure(df, title=str(args.stack_dir))
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({len(df)} objects)")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
