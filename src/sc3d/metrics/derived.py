"""Derived metrics computed from already-merged shape + intensity rows."""
from __future__ import annotations

import math

from ..core.stack import Stack
from .base import RowMap
from .registry import MetricGroup, register


@register(MetricGroup.DERIVED)
class DerivedMetrics:
    name = "derived"
    columns = ("centroid_displacement", "fluor_cv", "fluor_density")

    def compute(self, stack: Stack, prior: RowMap) -> RowMap:
        out: RowMap = {}
        for lbl, row in prior.items():
            gx, gy, gz = row.get("centroid_x"), row.get("centroid_y"), row.get("centroid_z")
            wx, wy, wz = (row.get("fluor_weighted_cx"), row.get("fluor_weighted_cy"),
                          row.get("fluor_weighted_cz"))

            if None in (gx, gy, gz, wx, wy, wz) or any(map(_isnan, (wx, wy, wz))):
                disp = float("nan")
            else:
                disp = math.sqrt((gx - wx) ** 2 + (gy - wy) ** 2 + (gz - wz) ** 2)

            mean = row.get("fluor_mean", float("nan"))
            std = row.get("fluor_std", float("nan"))
            cv = (std / mean) if (mean and not _isnan(mean) and mean != 0) else float("nan")

            volume = row.get("volume_phys", float("nan"))
            fsum = row.get("fluor_sum", float("nan"))
            density = (fsum / volume) if (volume and not _isnan(volume) and volume != 0) else float("nan")

            out[lbl] = {
                "centroid_displacement": disp,
                "fluor_cv": cv,
                "fluor_density": density,
            }
        return out


def _isnan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
