"""Default NIfTI loader."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import SimpleITK as sitk

from ..core.stack import Stack
from .base import StackRef
from .discovery import walk_segmentations


class NiftiStackLoader:
    def discover(self, root: Path) -> Iterator[StackRef]:
        return walk_segmentations(root)

    def load(self, ref: StackRef) -> Stack:
        seg = sitk.ReadImage(str(ref.seg_path))
        fluor = sitk.ReadImage(str(ref.fluor_path)) if ref.fluor_path.exists() else None

        if fluor is not None:
            # Spec §6: fluor and label map must share physical space.
            # Sizes/origins can drift after padding upstream — resample fluor
            # onto the segmentation grid (nearest-neighbor preserves raw
            # intensities and avoids interpolation bias on integer ADUs).
            if (seg.GetSize() != fluor.GetSize()
                    or seg.GetOrigin() != fluor.GetOrigin()
                    or seg.GetSpacing() != fluor.GetSpacing()
                    or seg.GetDirection() != fluor.GetDirection()):
                fluor = sitk.Resample(
                    fluor,
                    seg,
                    sitk.Transform(),
                    sitk.sitkNearestNeighbor,
                    0,
                    fluor.GetPixelID(),
                )

        return Stack(
            sample=ref.sample,
            segmentation=seg,
            fluor=fluor,
            spacing=tuple(seg.GetSpacing()),
        )
