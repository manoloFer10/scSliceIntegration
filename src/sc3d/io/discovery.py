"""Walk a Day_X/Condition/stack tree and find segmentation files.

Handles both uncompressed `.nii` (the original NIfTIs/ tree) and gzip-compressed
`.nii.gz` (the Leica converter output) — discovery must match either so the same
pipeline runs over both trees.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .. import config
from .base import StackRef


def _strip_base_ext(name: str, base: str) -> str | None:
    """Return the stem of `name` if it ends with `base` + a known NIfTI extension.

    Tries extensions longest-first (`.nii.gz` before `.nii`) so the compressed
    suffix is stripped whole. Returns None when `name` is not a `base` file.
    """
    for ext in config.NII_EXTS:
        suffix = f"{base}{ext}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def walk_segmentations(root: Path) -> Iterator[StackRef]:
    """Yield a StackRef for every `*_segmentation.nii[.gz]` under root.

    The paired fluor is `{stem}_fluor.nii[.gz]` — preferring the segmentation's own
    extension, then the other. When no fluor exists, `fluor_path` points at the
    default (non-existent) path; the loader tolerates that via `.exists()`.

    `sample` is the path of the stack folder relative to `root`, using forward
    slashes — stable across OSes and matches the existing CSV convention.
    """
    root = root.resolve()
    seen: set[tuple[Path, str]] = set()

    for seg_path in sorted(root.rglob(f"*{config.SEG_BASE}.nii*")):
        stem = _strip_base_ext(seg_path.name, config.SEG_BASE)
        if stem is None:
            continue  # e.g. a stray "*_segmentation.niigz" — not a real NIfTI
        stack_dir = seg_path.parent

        key = (stack_dir, stem)
        if key in seen:  # a stack carrying both .nii and .nii.gz yields once
            continue
        seen.add(key)

        seg_ext = seg_path.name[len(stem) + len(config.SEG_BASE):]  # ".nii" or ".nii.gz"
        fluor_exts = (seg_ext, *(e for e in config.NII_EXTS if e != seg_ext))
        fluor_path = stack_dir / f"{stem}{config.FLUOR_BASE}{seg_ext}"  # default (may not exist)
        for ext in fluor_exts:
            candidate = stack_dir / f"{stem}{config.FLUOR_BASE}{ext}"
            if candidate.exists():
                fluor_path = candidate
                break

        sample = stack_dir.relative_to(root).as_posix()
        yield StackRef(
            sample=sample,
            stack_dir=stack_dir,
            seg_path=seg_path,
            fluor_path=fluor_path,
        )
