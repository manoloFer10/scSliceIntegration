"""Manual black-out patches for spatially fixed, threshold-inseparable artifacts.

Some Leica fields carry detector haze / low-frequency noise concentrated in fixed
image regions (e.g. the bottom corners of the COR Day-3 series) that survives the
absolute GFP threshold and fragments into many size-passing components, inflating
the object count (see plans/preprocessing_log/2026-07-01-count-run-r1.md, where
`COR/Day_3/Series4` counts 236 vs a ~54-100 field range). When the artifact is
spatially fixed and *not* separable by the global threshold, the pragmatic fix is
to zero those regions before segmentation so no component can form there.

This module rasterizes user-specified convex polygons — given in image (x, y)
pixel coordinates, origin top-left, y increasing downward — and zeros them across
the full z-axis of a volume. It is deliberately generic: the *caller* decides
which series get which patches (only COR/Day_3 in the current re-render).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk

Polygon = Sequence[Sequence[float]]  # ordered [(x, y), ...] vertices

# Two black-out patches covering the noisy bottom corners of the COR Day-3 fields,
# in (x, y) pixel coordinates (origin top-left, y downward). Verified against the
# existing r1 segmentation of COR/Day_3/Series4: these regions hold the excess
# ConnectedComponent centroids (5 bottom-left + 152 bottom-right of 236) that
# inflate its count. Applied across all z-slices. Used only for the COR/Day_3
# series in the corner-masked re-render (see render_cor_masked.py).
COR_DAY3_PATCHES: tuple[Polygon, ...] = (
    ((54, 957), (0, 957), (0, 1024), (57, 1024)),          # bottom-left
    ((807, 876), (807, 1024), (1024, 1024), (1024, 876)),  # bottom-right
)


def polygon_mask(shape_yx: tuple[int, int], polygon_xy: Polygon) -> np.ndarray:
    """Rasterize a convex polygon to a boolean (Y, X) mask (True = inside).

    Vertices are (x, y): x the column, y the row, origin top-left. A pixel centre
    is inside when it lies on the same side of every directed edge (convex-polygon
    half-plane test), inclusive of the boundary; vertices may be given in either
    winding order. Computation is restricted to the polygon bounding box, so a
    small patch on a 1024² grid touches only its own footprint.

    A vertex coordinate of `N` (e.g. 1024 on a 1024-wide axis) is the exclusive
    far edge: pixel indices run 0..N-1, so the last covered column/row is N-1.
    """
    Y, X = shape_yx
    verts = np.asarray(polygon_xy, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        raise ValueError(f"polygon must be >=3 (x, y) vertices, got shape {verts.shape}")

    x0 = max(0, int(np.floor(verts[:, 0].min())))
    x1 = min(X, int(np.ceil(verts[:, 0].max())))
    y0 = max(0, int(np.floor(verts[:, 1].min())))
    y1 = min(Y, int(np.ceil(verts[:, 1].max())))
    mask = np.zeros((Y, X), dtype=bool)
    if x0 >= x1 or y0 >= y1:
        return mask

    ys, xs = np.mgrid[y0:y1, x0:x1]
    px = xs.ravel().astype(float)
    py = ys.ravel().astype(float)
    all_nonneg = np.ones(px.shape, dtype=bool)
    all_nonpos = np.ones(px.shape, dtype=bool)
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        all_nonneg &= cross >= -1e-9
        all_nonpos &= cross <= 1e-9
    inside = all_nonneg | all_nonpos
    mask[y0:y1, x0:x1] = inside.reshape(ys.shape)
    return mask


def apply_black_patches(image: sitk.Image, polygons_xy: Sequence[Polygon]) -> sitk.Image:
    """Return a copy of `image` with each polygon zeroed across the full z-axis.

    `image` is a 3-D (Z, Y, X) SimpleITK volume; each polygon is a sequence of
    (x, y) pixel vertices in the XY plane. The union of the polygons is set to 0
    on every z-slice. Spacing/origin/direction are preserved via CopyInformation.
    Returns the input unchanged (a copy) when `polygons_xy` is empty.
    """
    arr = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {arr.shape}")
    _, Y, X = arr.shape

    union = np.zeros((Y, X), dtype=bool)
    for poly in polygons_xy:
        union |= polygon_mask((Y, X), poly)

    out_arr = arr.copy()
    out_arr[:, union] = 0
    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(image)
    return out
