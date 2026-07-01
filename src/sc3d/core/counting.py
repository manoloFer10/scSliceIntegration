"""Count 3-D objects per stack in a preprocessed NIfTI tree.

This is the counting-only path: it walks a tree of segmentation masks, counts the
3-D connected components in each, and writes one `sample,n_objects` row per stack.
It deliberately does **not** load the fluor channel or run any shape/intensity/
derived metric provider — the deliverable is the GFP+ cell count and nothing else.

The segmentation masks written by `preprocess.leica` are already background-separated
and physical-size-filtered (`segment_fluorescence`, `min_volume_um3`), so re-running
connected components here reproduces the converter's count exactly. `MIN_VOXELS` is a
harmless floor on such a mask (verified: raw CC == floored CC on the r1 tree).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from .. import config
from ..io.discovery import walk_segmentations
from .labeling import label_and_filter
from .stack import Stack

log = logging.getLogger(__name__)


def count_stack(seg_path: Path, sample: str, min_voxels: int = config.MIN_VOXELS) -> int:
    """Return the number of 3-D connected components in a segmentation file."""
    seg = sitk.ReadImage(str(seg_path))
    stack = Stack(sample=sample, segmentation=seg, fluor=None, spacing=tuple(seg.GetSpacing()))
    _, counts = label_and_filter(stack, min_voxels=min_voxels)  # 26-conn CC + size floor
    return len(counts)


def count_tree(
    root: Path,
    out_dir: Path | None = None,
    min_voxels: int = config.MIN_VOXELS,
) -> Path:
    """Count objects for every stack under `root`; write `all_samples_counts.csv`.

    The CSV (`sample,n_objects`, one row per stack, sorted by sample) is written to
    `out_dir` (defaults to `root`). Per-stack failures are logged and skipped so one
    unreadable mask does not abort the run. Returns the output path.
    """
    out_dir = out_dir or root
    rows: list[dict] = []

    for ref in walk_segmentations(root):
        log.info("Counting %s", ref.sample)
        try:
            n = count_stack(ref.seg_path, ref.sample, min_voxels=min_voxels)
        except Exception:
            log.exception("Failed to count %s", ref.sample)
            continue
        rows.append({"sample": ref.sample, "n_objects": n})

    out_path = out_dir / config.COUNTS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["sample", "n_objects"])
    if not df.empty:
        df = df.sort_values("sample", kind="stable").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %d count rows to %s", len(df), out_path)
    return out_path
