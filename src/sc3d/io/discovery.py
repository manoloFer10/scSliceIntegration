"""Walk a Day_X/Condition/stack tree and find segmentation files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .. import config
from .base import StackRef


def walk_segmentations(root: Path) -> Iterator[StackRef]:
    """Yield a StackRef for every `*_segmentation.nii` under root.

    `sample` is the path of the stack folder relative to `root`, using forward
    slashes — stable across OSes and matches the existing CSV convention.
    """
    root = root.resolve()
    for seg_path in root.rglob(f"*{config.SEG_SUFFIX}"):
        stack_dir = seg_path.parent
        stem = seg_path.name[: -len(config.SEG_SUFFIX)]
        fluor_path = stack_dir / f"{stem}{config.FLUOR_SUFFIX}"
        sample = stack_dir.relative_to(root).as_posix()
        yield StackRef(
            sample=sample,
            stack_dir=stack_dir,
            seg_path=seg_path,
            fluor_path=fluor_path,
        )
