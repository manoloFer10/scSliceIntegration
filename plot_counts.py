#!/usr/bin/env python
"""Plot GFP+ 3D object counts per day, one panel per tissue.

Standalone reporting script (Plan 3, `plans/03-output-format.md`). It is run
*independently* of the `sc3d` pipeline: it reads an already-written counts CSV and
renders a figure. It does **not** re-run counting and does **not** write any CSV.

Input CSV
---------
Either of the two shapes the pipeline emits is accepted:
  * `all_samples_counts.csv`  — columns `sample,n_objects` (one row per series;
    the direct output of `sc3d count`). Used as-is.
  * `all_samples_metrics.csv` — the per-object aggregate (one row per labeled
    object). The per-series count is recovered as `groupby("sample").size()`.

`sample` is `"<TISSUE>/Day_N/<Series>"` (e.g. `COR/Day_3/Series4`); it is parsed
into tissue / day / series.

Excluded field
--------------
`COR/Day_3/Series4` (n=236) is a noisy-bottom-corners acquisition artifact
(detector haze that survives the absolute threshold and fragments into many
size-passing clusters), not a biological count. It is dropped by default; pass
`--keep-outliers` to include it.

Caveat (also stamped on the figure)
-----------------------------------
Absolute counts are threshold-method-dependent: confluent cells are undercounted
and deep signal is attenuated. Each day is an *independent* set of specimens, not
the same specimen tracked over time, so cross-day **trends** are more trustworthy
than the absolute magnitudes.

Usage
-----
    python plot_counts.py                                   # defaults below
    python plot_counts.py miscelaneous/all_samples_counts.csv
    python plot_counts.py <counts.csv> --out fig.png --show
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; overridden below only if --show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Field excluded as a data-quality artifact (see module docstring /
# plans/preprocessing_log/2026-07-01-count-run-r1.md).
DEFAULT_EXCLUDE: tuple[str, ...] = ("COR/Day_3/Series4",)

CAVEAT = (
    "Counts = absolute threshold + 3D connected components: confluent cells are "
    "undercounted and deep signal is depth-attenuated. Each day is an independent "
    "set of specimens (not the same specimen over time), so cross-day trends are "
    "more trustworthy than absolute magnitudes."
)

_DAY_RE = re.compile(r"Day[_\s]*(\d+)", re.IGNORECASE)


def _parse_day(day_str: str) -> int:
    m = _DAY_RE.search(day_str)
    if not m:
        raise ValueError(f"cannot parse day from {day_str!r}")
    return int(m.group(1))


def load_counts(csv_path: Path) -> pd.DataFrame:
    """Read a counts/metrics CSV → tidy `DataFrame[sample, tissue, day, series, count]`.

    Accepts the `sample,n_objects` counts CSV directly, or recovers per-series
    counts from a per-object metrics aggregate via `groupby(sample).size()`.
    """
    df = pd.read_csv(csv_path)
    if "sample" not in df.columns:
        raise ValueError(f"{csv_path}: no 'sample' column (columns={list(df.columns)})")

    if "n_objects" in df.columns:
        counts = df[["sample", "n_objects"]].rename(columns={"n_objects": "count"})
    else:  # per-object aggregate: one row per object → count = rows per sample
        counts = (
            df.groupby("sample", sort=False).size().rename("count").reset_index()
        )

    parts = counts["sample"].str.split("/", expand=True)
    if parts.shape[1] < 3:
        raise ValueError(
            "sample values must look like '<TISSUE>/Day_N/<Series>'; "
            f"got e.g. {counts['sample'].iloc[0]!r}"
        )
    counts["tissue"] = parts[0]
    counts["day"] = parts[1].map(_parse_day)
    counts["series"] = parts[2]
    return counts[["sample", "tissue", "day", "series", "count"]]


def _print_summary(df: pd.DataFrame) -> None:
    """Text-only per tissue×day summary (n / mean / median / min / max)."""
    print(f"{'tissue':>7} {'day':>4} {'n':>3} {'mean':>7} {'median':>7} {'min':>4} {'max':>4}")
    for (tissue, day), g in df.groupby(["tissue", "day"]):
        print(
            f"{tissue:>7} {day:>4} {len(g):>3} "
            f"{g['count'].mean():>7.1f} {g['count'].median():>7.1f} "
            f"{int(g['count'].min()):>4} {int(g['count'].max()):>4}"
        )


def plot_counts(df: pd.DataFrame, out_path: Path, excluded: list[str]) -> Path:
    """One panel per tissue: per-day box (spread over series) + jittered strip + mean trend."""
    tissues = sorted(df["tissue"].unique())
    days = sorted(df["day"].unique())
    x = np.arange(len(days))
    rng = np.random.default_rng(0)  # reproducible strip jitter

    fig, axes = plt.subplots(
        1, len(tissues), figsize=(5.0 * len(tissues) + 1, 5.4),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, tissue in zip(axes, tissues):
        sub = df[df["tissue"] == tissue]
        per_day = [sub.loc[sub["day"] == d, "count"].to_numpy() for d in days]

        bp = ax.boxplot(
            per_day, positions=x, widths=0.55, patch_artist=True,
            showfliers=False, medianprops=dict(color="#37474f", lw=1.4),
            whiskerprops=dict(color="#90a4ae"), capprops=dict(color="#90a4ae"),
            boxprops=dict(edgecolor="#90a4ae"),
        )
        for patch in bp["boxes"]:
            patch.set(facecolor="#e3f2fd", alpha=0.8)

        # jittered strip of the individual series counts
        for xi, vals in zip(x, per_day):
            if len(vals):
                jitter = rng.uniform(-0.13, 0.13, size=len(vals))
                ax.scatter(
                    xi + jitter, vals, s=34, color="#1565c0",
                    edgecolor="white", linewidth=0.5, zorder=3, alpha=0.9,
                )

        # per-day mean, connected across days to show the trend
        means = [float(np.mean(v)) if len(v) else np.nan for v in per_day]
        ax.plot(x, means, "-", color="#c62828", lw=1.6, zorder=4)
        ax.plot(
            x, means, "D", color="#c62828", ms=8, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5, label="mean",
        )

        ax.set_title(tissue, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Day {d}" for d in days])
        ax.set_xlabel("Day")
        ax.margins(x=0.12)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.legend(loc="upper left", frameon=False, fontsize=9)

    axes[0].set_ylabel("Cell count")
    fig.suptitle("Fluorescent cell counts per day", fontsize=14, fontweight="bold")

    note = CAVEAT
    if excluded:
        note += "  Excluded: " + ", ".join(excluded) + " (noisy-corner artifact)."
    #fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=7.5,
    #         color="#555555", wrap=True)

    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="?",
                    default=Path("miscelaneous/all_samples_counts.csv"),
                    help="Counts CSV (sample,n_objects) or per-object metrics aggregate "
                         "(default: miscelaneous/all_samples_counts.csv).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG (default: <csv dir>/cell_counts_per_day.png).")
    ap.add_argument("--keep-outliers", action="store_true",
                    help=f"Do not drop the excluded field(s): {', '.join(DEFAULT_EXCLUDE)}.")
    ap.add_argument("--show", action="store_true", help="Also display the figure window.")
    args = ap.parse_args(argv)

    df = load_counts(args.csv)

    excluded: list[str] = []
    if not args.keep_outliers:
        mask = df["sample"].isin(DEFAULT_EXCLUDE)
        excluded = df.loc[mask, "sample"].tolist()
        if excluded:
            print("Excluding (data-quality artifact): " + ", ".join(excluded))
        df = df[~mask].reset_index(drop=True)

    print(f"\nCAVEAT: {CAVEAT}\n")
    _print_summary(df)
    print()

    out_path = args.out or args.csv.parent / "cell_counts_per_day.png"
    if args.show:
        matplotlib.use("TkAgg", force=True)
    plot_counts(df, out_path, excluded)
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
