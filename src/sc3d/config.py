"""Pipeline configuration constants."""
from __future__ import annotations

SEG_SUFFIX = "_segmentation.nii"
FLUOR_SUFFIX = "_fluor.nii"
RAW_SUFFIX = "_raw.nii"

MIN_VOXELS = 10  # Spec §6: objects with < 10 voxels are noise.

PER_STACK_CSV = "cell_metrics_per_object.csv"
AGGREGATE_CSV = "all_samples_metrics.csv"
