"""GFP fluorescence preprocessing + segmentation.

Two stages, both operating on the ch00 GFP volume:

1. `remove_watermark` — zero the burned-in Leica scale-bar overlay (Impurity 1).
2. `segment_fluorescence` — XY median denoise → absolute threshold → 3-D
   connected components → physical-size filter (Impurity 2).

See `plans/preprocessing_log/2026-07-01-data-impurities.md` for the diagnosis and
the empirical operating-point sweep behind the defaults below.
"""
from __future__ import annotations

import math
import logging

import numpy as np
import SimpleITK as sitk

from sc3d import config

logger = logging.getLogger(__name__)


def remove_watermark(
    fluor: sitk.Image,
    *,
    saturation: float = 200.0,
    z_fraction: float = 0.5,
    dilate_radius: int = 2,
) -> sitk.Image:
    """Zero the burned-in Leica scale-bar watermark in a GFP volume.

    The watermark ("0 µm 100" scale bar) is a static overlay pinned near saturation
    (~255) at the *same* (x, y) on *every* z-slice (Impurity 1 in the preprocessing
    log). We detect it position-independently — no hard-coded bbox — as the (x, y)
    columns that exceed `saturation` in more than `z_fraction` of the z-slices,
    dilate the 2-D mask by `dilate_radius` px to catch the anti-aliased glyph edges,
    and set those columns to 0 across all z.

    Zeroing (rather than inpainting) matches the true GFP background (~0) and keeps
    the overlay out of *both* the segmentation and the downstream intensity metrics
    (`fluor_mean`/`fluor_sum`), which it would otherwise inflate. A volume with no
    watermark is returned unchanged (the detection mask is empty → no-op), so this is
    safe to call unconditionally.
    """
    arr = sitk.GetArrayFromImage(fluor)  # (Z, Y, X)
    if arr.ndim != 3:
        raise ValueError(f"remove_watermark expects a 3-D volume, got shape {arr.shape}")

    z_hits = (arr > saturation).mean(axis=0)  # (Y, X): fraction of slices saturated
    mask2d = z_hits > z_fraction
    n_core = int(mask2d.sum())
    if n_core == 0:
        return fluor  # no persistent saturated overlay → nothing to remove

    if dilate_radius > 0:
        dilated = sitk.BinaryDilate(
            sitk.GetImageFromArray(mask2d.astype(np.uint8)), [dilate_radius, dilate_radius]
        )
        mask2d = sitk.GetArrayFromImage(dilated).astype(bool)

    cleaned = arr.copy()
    cleaned[:, mask2d] = 0
    out = sitk.GetImageFromArray(cleaned)
    out.CopyInformation(fluor)  # restore spacing/origin/direction
    logger.info(
        "Watermark removed: %d core px (>%.0f in >%.0f%% of z) → %d px after dilation",
        n_core, saturation, 100 * z_fraction, int(mask2d.sum()),
    )
    return out


def segment_fluorescence(
    fluor: sitk.Image,
    *,
    threshold: float | None = config.COUNT_THRESHOLD,
    min_volume_um3: float = config.COUNT_MIN_VOLUME_UM3,
    min_voxels: int | None = None,
    denoise_radius: int = config.COUNT_DENOISE_RADIUS,
    fully_connected: bool = config.COUNT_FULLY_CONNECTED,
) -> sitk.Image:
    """Segment GFP foreground blobs from a fluorescence volume.

    Pipeline: optional XY median denoise → threshold → 3-D connected components →
    physical minimum-size filter → binary mask.

    **Denoising (`denoise_radius`).** The GFP channel is a sparse set of bright cells
    on a dense field of single-voxel salt-and-pepper detector noise (Impurity 2). A
    median (Z radius fixed to 1) removes that speckle while preserving the
    ~12 px-radius cell discs.

    **Thresholding (`threshold`).** An absolute lower bound on GFP intensity. Otsu
    (`threshold=None`, retained for completeness) is *wrong* for this data: the sparse,
    mostly-zero histogram is not bimodal, so Otsu lands in the noise floor (≈7) and
    produces millions of components (Impurity 2). The default 20 sits just above the
    noise floor (cleaned p99 ≈ 19) inside the median-stabilised plateau. This operating
    point sets the cell counts (the deliverable) and is provisional pending visual
    confirmation on the reference series.

    **Size filter (`min_voxels` / `min_volume_um3`).** The authoritative noise-rejection
    knob. When `min_voxels` is given it is used directly (a spacing-independent voxel
    count); otherwise `min_volume_um3` is converted to voxels via the image spacing
    (300 µm³ ≈ 46 voxels at 1.1353×1.1353×5 µm). Objects smaller than the resulting
    voxel count are dropped.

    **Background subtraction (`bg_radius`).** Optional anisotropic white top-hat (large
    XY, thin Z). Off by default: on the confirmed Leica data it is ~inert (cells are
    smaller than the kernel and the true background is ≈0), so it only adds cost. Kept
    for stacks with a genuine low-frequency haze.

    Caveats:
    - Confluent/touching cells merge into one component → undercounting. Accepted;
      watershed is a future drop-in.
    - A single global threshold undercounts dim cells deep in the stack (GFP decays
      with depth). The plateau suggests the effect is small here.
    - `min_volume_um3` relies on correct image spacing; wrong spacing → wrong filter.
    - Thresholding assumes an 8-bit (0–255) intensity range.
    """
    if min_voxels is not None:
        min_voxels = max(1, int(min_voxels))
    else:
        spacing = fluor.GetSpacing()  # (sx, sy, sz) in µm
        voxel_um3 = math.prod(spacing)
        min_voxels = max(1, round(min_volume_um3 / voxel_um3))

    work = fluor

    # XY-only median denoise. Radius in voxels: [r, r, 1].
    if denoise_radius and denoise_radius > 0:
        median = sitk.MedianImageFilter()
        median.SetRadius([denoise_radius, denoise_radius, 1])
        work = median.Execute(work)

    if threshold is None:
        # ITK OtsuThreshold assigns insideValue BELOW the threshold; foreground=1 wants
        # (insideValue=0, outsideValue=1).
        mask = sitk.OtsuThreshold(work, 0, 1)
    else:
        # this is default behavior intended.
        mask = sitk.BinaryThreshold(
            work, lowerThreshold=float(threshold), upperThreshold=255.0,
            insideValue=1, outsideValue=0,
        )

    cc = sitk.ConnectedComponent(mask, fully_connected)
    relabeled = sitk.RelabelComponent(cc, minimumObjectSize=min_voxels, sortByObjectSize=True)

    # Binarize: any surviving label → 1
    binary = sitk.Cast(relabeled > 0, sitk.sitkUInt8)
    binary.CopyInformation(fluor)
    return binary
