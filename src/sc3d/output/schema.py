"""Canonical column order for output CSVs.

Mirrors NIfTIs/all_samples_metrics.csv so downstream consumers don't break.
"""
from __future__ import annotations

CANONICAL_COLUMNS: tuple[str, ...] = (
    "sample",
    "label",
    "n_voxels",
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
    "centroid_displacement",
    "fluor_cv",
    "fluor_density",
)
