"""Connected-component labeling, size-ordered relabel, and noise filtering."""
from __future__ import annotations

import SimpleITK as sitk

from .stack import Stack


def label_and_filter(stack: Stack, min_voxels: int) -> tuple[Stack, dict[int, int]]:
    """Re-label `stack.segmentation` so that:
      - connected components are 1-indexed,
      - labels are sorted by descending voxel count (label 1 = largest),
      - components with < `min_voxels` voxels are dropped.

    Returns the updated Stack and a {label: n_voxels} map for the kept labels.
    """
    seg = stack.segmentation
    # Treat any non-zero as foreground; CC then re-labels.
    binary = sitk.BinaryThreshold(seg, lowerThreshold=1, upperThreshold=2**31 - 1,
                                   insideValue=1, outsideValue=0)
    cc = sitk.ConnectedComponent(binary, True)  # 26-connectivity, matching segment_fluorescence
    relabel = sitk.RelabelComponent(cc, minimumObjectSize=min_voxels, sortByObjectSize=True)

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(relabel)
    counts = {int(lbl): int(shape_stats.GetNumberOfPixels(int(lbl)))
              for lbl in shape_stats.GetLabels()}

    new_stack = Stack(
        sample=stack.sample,
        segmentation=relabel,
        fluor=stack.fluor,
        spacing=stack.spacing,
    )
    return new_stack, counts
