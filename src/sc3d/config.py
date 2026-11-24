"""Pipeline configuration constants."""
from __future__ import annotations

# Stack file naming. Files may be uncompressed (`.nii`, the original NIfTIs/ tree)
# or gzip-compressed (`.nii.gz`, the Leica converter output). Discovery matches both.
SEG_BASE = "_segmentation"
FLUOR_BASE = "_fluor"
RAW_BASE = "_raw"
NII_EXTS = (".nii.gz", ".nii")  # longest-match first (see io/discovery)

# Back-compat aliases (uncompressed variants); prefer the *_BASE + NII_EXTS forms.
SEG_SUFFIX = f"{SEG_BASE}.nii"
FLUOR_SUFFIX = f"{FLUOR_BASE}.nii"
RAW_SUFFIX = f"{RAW_BASE}.nii"

MIN_VOXELS = 10  # Spec §6: objects with < 10 voxels are noise.

PER_STACK_CSV = "cell_metrics_per_object.csv"
AGGREGATE_CSV = "all_samples_metrics.csv"
COUNTS_CSV = "all_samples_counts.csv"

# --- Leica confocal converter + GFP counting (see plans/01, plans/02 and
# plans/preprocessing_log/2026-07-01-data-impurities.md). Single source of truth
# for the segmentation hyperparameters; the "r1" operating point. ---
COUNT_CHANNEL = "ch00"   # green-LUT GFP, the countable channel
RAW_CHANNEL = "ch01"     # uniform brightfield, not countable (kept as *_raw)
# Real physical spacing (µm); the TIF-embedded 0.35278 µm is a bogus 1/72-inch default.
LEICA_SPACING_UM = (1.1353, 1.1353, 5.0)  # (sx, sy, sz)
COUNT_THRESHOLD = 20.0        # absolute GFP intensity floor (just above the noise p99)
COUNT_MIN_VOLUME_UM3 = 300.0  # object-size filter in physical units (~46 vox at r1 spacing).
                              # Used only when COUNT_MIN_VOXELS is None (spacing-derived path).
COUNT_MIN_VOXELS = 60         # authoritative object-size filter as a DIRECT voxel count
                              # (spacing-independent). Takes precedence over COUNT_MIN_VOLUME_UM3.
COUNT_DENOISE_RADIUS = 1      # XY(+Z=1) median radius that creates the count plateau
COUNT_FULLY_CONNECTED = True  # 26-connectivity, matching core/labeling
COUNT_IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg")  # per-slice image formats the converter reads
LEICA_OUT_DIRNAME = "NIfTIs_leica"
# Tissue folder: "Analysis_COR (Cortex)", "Analysis_WER (Epidermis)", "Analysis_Col0" → COR/WER/Col0.
LEICA_TISSUE_RE = r"Analysis_(\w+)"
# Day folder: matches both "Analysis_Day 3" and the compact "Analysis_D3" → Day_{n}.
LEICA_DAY_RE = r"Analysis_D(?:ay)?\s*(\d+)"
