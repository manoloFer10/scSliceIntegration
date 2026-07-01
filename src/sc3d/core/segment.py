"""GFP fluorescence segmentation: background subtraction → threshold → connected components."""
from __future__ import annotations

import math
import logging

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def segment_fluorescence(
    fluor: sitk.Image,
    *,
    threshold: float | None = None,
    min_volume_um3: float = 50.0,
    bg_radius: int = 15,
) -> sitk.Image:
    """Segment GFP foreground blobs from a fluorescence volume.

    Background subtraction uses an anisotropic white top-hat with a box kernel
    whose XY radius equals bg_radius but whose Z radius is 1 (one voxel).  This
    removes the low-frequency depth-attenuation haze that confocal GFP stacks
    accumulate along Z without blurring axial boundaries.  The price is that any
    structure wider than 2*bg_radius µm in XY will be partially suppressed — keep
    bg_radius well below the expected cell diameter.

    Thresholding is Otsu on the top-hat image when threshold is None, otherwise a
    hard lower bound.  Connected-component analysis then filters objects smaller
    than min_volume_um3 (physical units).

    Caveats:
    - Confluent or touching cells merge into one component → undercounting.
      The plan notes this is accepted; watershed is a future drop-in.
    - Depth attenuation in the raw data is only partially compensated.  Dim cells
      near the bottom of the stack may fall below the Otsu threshold.
    - min_volume_um3 is converted to voxels using the image spacing; if the image
      spacing is wrong the filter threshold will be wrong.
    """
    spacing = fluor.GetSpacing()  # (sx, sy, sz) in µm
    voxel_um3 = math.prod(spacing)
    min_voxels = max(1, round(min_volume_um3 / voxel_um3))

    # Anisotropic white top-hat: large XY footprint, thin in Z.
    # KernelRadius is interpreted as half-width in each dimension.
    f_tophat = sitk.WhiteTopHatImageFilter()
    f_tophat.SetKernelType(sitk.sitkBox)
    xy_r = bg_radius
    z_r = 1
    f_tophat.SetKernelRadius([xy_r, xy_r, z_r])
    tophat = f_tophat.Execute(fluor)

    if threshold is None:
        # ITK OtsuThreshold convention: insideValue is assigned to voxels BELOW the
        # threshold (background).  We want foreground=1, so pass (insideValue=0, outsideValue=1).
        mask = sitk.OtsuThreshold(tophat, 0, 1)
    else:
        mask = sitk.BinaryThreshold(tophat, lowerThreshold=float(threshold), upperThreshold=255.0,
                                    insideValue=1, outsideValue=0)

    cc = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(cc, minimumObjectSize=min_voxels, sortByObjectSize=True)

    # Binarize: any non-zero label → 1
    binary = sitk.Cast(relabeled > 0, sitk.sitkUInt8)
    binary.CopyInformation(fluor)
    return binary
