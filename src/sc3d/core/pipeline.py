"""End-to-end orchestration: discover → load → label → metrics → write."""
from __future__ import annotations

import logging
from pathlib import Path

from .. import config, metrics  # imports register all providers
from ..io.base import StackLoader, StackRef
from ..output.writer import write_aggregate, write_per_stack
from .labeling import label_and_filter
from .stack import Stack

log = logging.getLogger(__name__)


def compute_all(stack: Stack, n_voxels: dict[int, int]) -> list[dict]:
    """Run every registered provider in fixed group order, merge into rows."""
    merged: dict[int, dict[str, float]] = {lbl: {} for lbl in n_voxels}
    for group in metrics.GROUP_ORDER:
        for provider in metrics.providers(group):
            result = provider.compute(stack, merged)
            for lbl, cols in result.items():
                if lbl in merged:
                    merged[lbl].update(cols)

    rows: list[dict] = []
    for lbl in sorted(merged):  # label 1 = largest, ascending
        row = {"sample": stack.sample, "label": lbl, "n_voxels": n_voxels[lbl]}
        row.update(merged[lbl])
        rows.append(row)
    return rows


def process_stack(loader: StackLoader, ref: StackRef) -> list[dict]:
    stack = loader.load(ref)
    stack, n_voxels = label_and_filter(stack, min_voxels=config.MIN_VOXELS)
    return compute_all(stack, n_voxels)


def run(loader: StackLoader, root: Path, aggregate_dir: Path | None = None) -> Path:
    """Walk `root`, process every stack, and write outputs.

    Per-stack CSVs are written next to each stack's segmentation file.
    The aggregated CSV is written to `aggregate_dir` (defaults to `root`).
    """
    aggregate_dir = aggregate_dir or root
    aggregate_rows: list[dict] = []

    for ref in loader.discover(root):
        log.info("Processing %s", ref.sample)
        try:
            rows = process_stack(loader, ref)
        except Exception:
            log.exception("Failed to process %s", ref.sample)
            continue
        write_per_stack(rows, ref.stack_dir / config.PER_STACK_CSV)
        aggregate_rows.extend(rows)

    out_path = aggregate_dir / config.AGGREGATE_CSV
    write_aggregate(aggregate_rows, out_path)
    log.info("Wrote %d rows to %s", len(aggregate_rows), out_path)
    return out_path
