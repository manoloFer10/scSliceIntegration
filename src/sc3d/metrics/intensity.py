"""Fluorescence intensity metrics from LabelIntensityStatisticsImageFilter.

Median is not exposed by SimpleITK's filter, so it's computed per-label from
the masked numpy view.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from ..core.stack import Stack
from .base import RowMap
from .registry import MetricGroup, register


@register(MetricGroup.INTENSITY)
class IntensityMetrics:
    name = "intensity"
    columns = (
        "fluor_mean",
        "fluor_median",
        "fluor_std",
        "fluor_min",
        "fluor_max",
        "fluor_sum",
        "fluor_kurtosis",
        "fluor_skewness",
        "fluor_weighted_cx",
        "fluor_weighted_cy",
        "fluor_weighted_cz",
    )

    def compute(self, stack: Stack, prior: RowMap) -> RowMap:
        if stack.fluor is None:
            return {lbl: {c: float("nan") for c in self.columns} for lbl in prior}

        f = sitk.LabelIntensityStatisticsImageFilter()
        f.Execute(stack.segmentation, stack.fluor)

        seg_arr = sitk.GetArrayViewFromImage(stack.segmentation)
        flu_arr = sitk.GetArrayViewFromImage(stack.fluor)

        out: RowMap = {}
        for lbl in f.GetLabels():
            lbl = int(lbl)
            cx, cy, cz = f.GetCenterOfGravity(lbl)
            mask = seg_arr == lbl
            vals = flu_arr[mask]
            median = float(np.median(vals)) if vals.size else 0.0
            out[lbl] = {
                "fluor_mean": float(f.GetMean(lbl)),
                "fluor_median": median,
                "fluor_std": float(f.GetStandardDeviation(lbl)),
                "fluor_min": float(f.GetMinimum(lbl)),
                "fluor_max": float(f.GetMaximum(lbl)),
                "fluor_sum": float(f.GetSum(lbl)),
                "fluor_kurtosis": float(f.GetKurtosis(lbl)),
                "fluor_skewness": float(f.GetSkewness(lbl)),
                "fluor_weighted_cx": float(cx),
                "fluor_weighted_cy": float(cy),
                "fluor_weighted_cz": float(cz),
            }
        return out
