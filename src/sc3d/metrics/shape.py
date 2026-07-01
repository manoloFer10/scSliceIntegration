"""Shape metrics from SimpleITK's LabelShapeStatisticsImageFilter."""
from __future__ import annotations

import math

import SimpleITK as sitk

from ..core.stack import Stack
from .base import RowMap
from .registry import MetricGroup, register


@register(MetricGroup.SHAPE)
class ShapeMetrics:
    name = "shape"
    columns = (
        "volume_phys",
        "surface_area_phys",
        "sphericity",
        "roundness",
        "elongation",
        "flatness",
        "equiv_spherical_radius",
        "surface_to_volume_ratio",
        "centroid_x",
        "centroid_y",
        "centroid_z",
    )

    def compute(self, stack: Stack, prior: RowMap) -> RowMap:
        f = sitk.LabelShapeStatisticsImageFilter()
        f.ComputePerimeterOn()
        f.ComputeFeretDiameterOff()
        f.ComputeOrientedBoundingBoxOff()
        f.Execute(stack.segmentation)

        out: RowMap = {}
        for lbl in f.GetLabels():
            lbl = int(lbl)
            volume = float(f.GetPhysicalSize(lbl))
            area = float(f.GetPerimeter(lbl))  # 3D: surface area.
            r_eq = float(f.GetEquivalentSphericalRadius(lbl))
            sphericity = (
                (math.pi ** (1 / 3)) * ((6.0 * volume) ** (2 / 3)) / area
                if area > 0 else 0.0
            )
            cx, cy, cz = f.GetCentroid(lbl)
            out[lbl] = {
                "volume_phys": volume,
                "surface_area_phys": area,
                "sphericity": sphericity,
                "roundness": float(f.GetRoundness(lbl)),
                "elongation": float(f.GetElongation(lbl)),
                "flatness": float(f.GetFlatness(lbl)),
                "equiv_spherical_radius": r_eq,
                "surface_to_volume_ratio": area / volume if volume > 0 else 0.0,
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "centroid_z": float(cz),
            }
        return out
