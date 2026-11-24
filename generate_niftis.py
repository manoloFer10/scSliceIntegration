"""
generate_niftis.py
------------------
Converts a z-stack image folder (containing _zXX_chYY.jpg slices) into up to
three NIfTI-1 volumes for visualization in ITK-SNAP:

    Cell counting/NIfTIs/Day_{N}/{condition}/{stack_name}/{stack_name}_raw.nii
    Cell counting/NIfTIs/Day_{N}/{condition}/{stack_name}/{stack_name}_fluor.nii
    Cell counting/NIfTIs/Day_{N}/{condition}/{stack_name}/{stack_name}_overlay.nii

The overlay NIfTI is an RGB volume with the raw channel in grayscale and the
fluorescence signal highlighted in green.  When a z-slice has no corresponding
fluorescence image, the raw image is included as plain grayscale.

Conventions
-----------
- ch00 → raw channel
- ch01 → fluorescence (green) channel
- Slices are sorted numerically by z-index extracted from filename suffix _zXX_
- Output volumes have shape (X, Y, Z) = (width, height, n_slices), which is the
  standard NIfTI voxel-grid orientation expected by ITK-SNAP.
- Affine is set to a 1 µm isotropic identity transform as a placeholder; update
  with real voxel calibration values once available (xy_pixel_um, z_step_um).

Usage
-----
From the command line (process a single stack):

    python generate_niftis.py "Cell counting/In vitro experiment/Day 1/Data for counting_Alginate_Day 1/well 1_Alginate_z stack 1"

Or import and call programmatically:

    from generate_niftis import generate_nifti
    generate_nifti(Path("..."))

To batch-process every stack across all days:

    python generate_niftis.py --all
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("generate_niftis")


# Regex: match the z-index and channel suffix in every filename variant
#   e.g.  "..._z00_ch00.jpg"  →  z_idx=0, ch_idx=0
SLICE_PATTERN = re.compile(r"_z(\d+)_ch(\d+)\.jpg$", re.IGNORECASE)

CH_FLUOR = 0   # ch00 — green fluorescence
CH_RAW   = 1   # ch01 — brightfield / raw

WORKSPACE_ROOT = Path(__file__).parent
NIFTI_ROOT = WORKSPACE_ROOT / "Cell counting" / "NIfTIs"
EXPERIMENT_ROOT = WORKSPACE_ROOT / "Cell counting" / "In vitro experiment"


def _extract_day(stack_dir: Path) -> str:
    """
    Walk up the parents of *stack_dir* to find a folder whose name matches
    "Day <N>" (e.g. "Day 1", "Day 3").  Returns the number as a string.

    Raises ValueError if no such parent is found.
    """
    for parent in stack_dir.parents:
        m = re.fullmatch(r"Day\s+(\d+)", parent.name, re.IGNORECASE)
        if m:
            return m.group(1)
    raise ValueError(
        f"Could not determine day number from path: {stack_dir}\n"
        "Expected a parent directory named 'Day N' (e.g. 'Day 1')."
    )


# Ordered so PA-RC is checked before PA to avoid a substring match.
_CONDITION_PATTERN = re.compile(r"(Alginate|PA-RC|PA|Control)", re.IGNORECASE)
_CONDITION_CANONICAL = {
    "alginate": "Alginate",
    "pa-rc":    "PA-RC",
    "pa":       "PA",
    "control":  "Control",
}


def _extract_condition(stack_dir: Path) -> str:
    """
    Infer the experimental condition (Alginate, PA, PA-RC, Control) by
    searching the *direct parent* folder name for a known condition token.

    Falls back to "Unknown" with a warning if none is found, so that
    batch processing never aborts on an unexpected folder name.
    """
    parent_name = stack_dir.parent.name
    m = _CONDITION_PATTERN.search(parent_name)
    if m:
        return _CONDITION_CANONICAL[m.group(1).lower()]
    log.warning(
        "Could not determine condition from parent folder '%s' for stack '%s'. "
        "Placing under 'Unknown'.",
        parent_name, stack_dir.name,
    )
    return "Unknown"

def _remove_length_tag(image):
    patch = np.zeros((38, 186))
    image[5:43, 5:191] = patch
    return image


def _process_slices(
        slices: list[tuple[int, Path]], 
        channel: int):
    frames: list[np.ndarray] = []
    if channel == CH_RAW:
        for _, filepath in slices:
            img = imread(str(filepath))             
            img = (rgb2gray(img) * 255).astype(np.uint8) #scale to 255, rgb2gray collapses to [0,1]
            img = np.rot90(img)
            img = np.rot90(img)
            frames.append(img)
    elif channel == CH_FLUOR:
        for _, filepath in slices:
            img = imread(str(filepath))             
            img = img[:,:,1] #green channel
            img = np.rot90(img)
            img = np.rot90(img)
            img = _remove_length_tag(img)
            frames.append(img)
    return frames

def _load_channel_slices(
    stack_dir: Path,
    channel: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Collect all .jpg files in *stack_dir* that belong to *channel*, sort them
    numerically by z-index, and return:

        volume   : np.ndarray, shape (Z, H, W), dtype uint8 or uint16
        z_indices: list[int] of the z-indices that were loaded
        missing  : list[int] of z-indices absent from the expected sequence

    Warnings are emitted for gaps in the z-sequence; no exception is raised
    because this function is used for visualization output only.
    """
    # Gather (z_index, filepath) pairs for the requested channel
    slices: list[tuple[int, Path]] = []
    for f in stack_dir.iterdir():
        m = SLICE_PATTERN.search(f.name)
        if m and int(m.group(2)) == channel:
            slices.append((int(m.group(1)), f))

    if not slices:
        raise FileNotFoundError(
            f"No slices found for channel {channel:02d} in:\n  {stack_dir}"
        )

    # Sort by z-index
    slices.sort(key=lambda x: x[0])
    z_indices = [z for z, _ in slices]

    # Warn about gaps in the z-sequence
    expected = list(range(z_indices[0], z_indices[-1] + 1))
    missing  = sorted(set(expected) - set(z_indices))
    if missing:
        warnings.warn(
            f"Stack '{stack_dir.name}' ch{channel:02d}: "
            f"missing z-slices {missing}. "
            f"The NIfTI will be built from the {len(slices)} available slices only.",
            UserWarning,
            stacklevel=3,
        )
        log.warning(
            "ch%02d missing z-slices %s in %s", channel, missing, stack_dir.name
        )

    frames = _process_slices(slices, channel)

    # Validate that all frames have the same spatial dimensions
    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        raise AssertionError(
            f"Inconsistent slice dimensions in '{stack_dir.name}' ch{channel:02d}: "
            f"{shapes}. All slices must have the same (H, W)."
        )

    volume = np.transpose(np.stack(frames, axis=-1), axes=(1,0,2))   # (H, W, Z)
    return volume, z_indices, missing


def _build_affine(xy_pixel_um: float = 1.0, z_step_um: float = 1.0) -> np.ndarray:
    """
    Build a diagonal 4×4 NIfTI affine matrix.

    The voxel sizes default to 1 µm isotropic (placeholder).  Pass real
    calibration values from pipeline.yaml when available.

    NIfTI convention: affine maps voxel indices (i, j, k) → mm coordinates.
    Nibabel interprets axis 0 as X, axis 1 as Y, axis 2 as Z.
    Here axis 0,1 are the in-plane (X, Y) axes and axis 2 is Z.
    """
    s = xy_pixel_um * 1e-3   # µm → mm  (ITK-SNAP expects mm)
    z = z_step_um  * 1e-3
    affine = np.diag([s, s, z, 1.0])
    return affine


def generate_nifti_and_segmentation(
    stack_dir: str | Path,
    xy_pixel_um: float = 1.0,
    z_step_um:   float = 1.0,
) -> tuple[Path, Path | None, Path, dict]:
    """
    Generate NIfTI volumes from the image stack in *stack_dir*:

    1. ``{name}_raw.nii``     — raw (brightfield) channel  (always)
    2. ``{name}_fluor.nii``   — fluorescence channel       (when available)

    Parameters
    ----------
    stack_dir    : Path to the folder containing *_zXX_chYY.jpg images.
    xy_pixel_um  : In-plane pixel size in micrometres (placeholder = 1.0).
    z_step_um    : Z-step between slices in micrometres (placeholder = 1.0).

    Returns
    -------
    (raw_path, fluor_path, overlay_path, missing_info)
        raw_path     — absolute path of the raw-channel NIfTI
        fluor_path   — absolute path of the fluorescence NIfTI, or None
        missing_info — dict with keys "ch00" and "ch01", each a list of
                       missing z-indices (empty list when complete)
    """
    stack_dir = Path(stack_dir).resolve()
    if not stack_dir.is_dir():
        raise NotADirectoryError(f"stack_dir does not exist or is not a folder:\n  {stack_dir}")

    # --- Determine output directory -------------------------------------------
    day_num    = _extract_day(stack_dir)
    condition  = _extract_condition(stack_dir)
    stack_name = stack_dir.name        # folder name becomes the NIfTI stem
    out_dir    = NIFTI_ROOT / f"Day_{day_num}" / condition / stack_name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path     = out_dir / f"{stack_name}_raw.nii"
    fluor_path   = out_dir / f"{stack_name}_fluor.nii"

    # --- Load channels --------------------------------------------------------
    raw_vol, z_raw, missing_raw = _load_channel_slices(stack_dir, CH_RAW)

    try:
        fluor_vol, z_fluor, missing_fluor = _load_channel_slices(stack_dir, CH_FLUOR)
    except FileNotFoundError:
        log.warning(
            "No fluorescence channel found for '%s'. ",
            stack_name,
        )
        fluor_vol, z_fluor, missing_fluor = None, [], []

    # --- Build NIfTI images -------
    affine = _build_affine(xy_pixel_um, z_step_um)
    raw_nii = nib.Nifti1Image(raw_vol, affine)
    raw_nii.header.set_xyzt_units("mm")
    nib.save(raw_nii, str(raw_path))
    
    labels = np.zeros(fluor_vol.shape, dtype=np.uint8)
    labels[fluor_vol > 128] = 1
    seg_from_fluor_nii = nib.Nifti1Image(labels,affine)
    seg_from_fluor_nii.header.set_xyzt_units("mm")
    nib.save(seg_from_fluor_nii, str(fluor_path))
    returned_fluor_path: Path | None = fluor_path

    return raw_path, returned_fluor_path, {"ch00": missing_raw, "ch01": missing_fluor}


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def _iter_all_stacks() -> list[Path]:
    """
    Recursively find every leaf directory under EXPERIMENT_ROOT that contains
    at least one file matching the SLICE_PATTERN.
    """
    stacks: list[Path] = []
    for p in sorted(EXPERIMENT_ROOT.rglob("*")):
        if p.is_dir():
            has_slices = any(SLICE_PATTERN.search(f.name) for f in p.iterdir() if f.is_file())
            if has_slices:
                stacks.append(p)
    return stacks


def batch_generate_niftis(
    xy_pixel_um: float = 1.0,
    z_step_um:   float = 1.0,
) -> list[tuple[Path, Path | None, Path]]:
    """
    Discover and process every stack folder under
    'Cell counting/In vitro experiment' and write NIfTIs for all of them.

    After processing, writes a JSON summary of any missing slices to:
        Cell counting/NIfTIs/missing_slices_report_{timestamp}.json

    Returns a list of (raw_path, fluor_path | None, overlay_path) tuples.
    """
    stacks = _iter_all_stacks()
    log.info("Found %d stacks to process.", len(stacks))

    results      : list[tuple[Path, Path]] = []
    errors       : list[tuple[Path, Exception]] = []
    missing_report: list[dict] = []

    for stack in tqdm(stacks, desc='Processing stacks', unit='stack'):
        try:
            raw_path, fluor_path, missing_info = generate_nifti_and_segmentation(
                stack, xy_pixel_um, z_step_um
            )
            results.append((raw_path, fluor_path))

            has_missing = bool(missing_info["ch00"] or missing_info["ch01"])
            if has_missing:
                try:
                    day_num = _extract_day(stack)
                except ValueError:
                    day_num = "unknown"
                try:
                    condition = _extract_condition(stack)
                except Exception:
                    condition = "Unknown"
                missing_report.append({
                    "day":        f"Day_{day_num}",
                    "condition":  condition,
                    "stack":      stack.name,
                    "stack_path": str(stack),
                    "ch00_missing_z": missing_info["ch00"],
                    "ch01_missing_z": missing_info["ch01"],
                })
        except Exception as exc:
            log.error("FAILED '%s': %s", stack.name, exc)
            errors.append((stack, exc))

    log.info("Finished: %d succeeded, %d failed.", len(results), len(errors))
    if errors:
        log.warning("Failed stacks:")
        for stack, exc in errors:
            log.warning("  %s — %s", stack, exc)

    # --- Write missing-slices JSON report ------------------------------------
    NIFTI_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = NIFTI_ROOT / f"missing_slices_report_{timestamp}.json"

    report_data = {
        "generated_at":       datetime.datetime.now().isoformat(),
        "total_stacks":       len(stacks),
        "succeeded":          len(results),
        "failed":             len(errors),
        "stacks_with_missing_slices": len(missing_report),
        "missing_slices":     missing_report,
        "failed_stacks": [
            {"stack": str(s), "error": str(e)} for s, e in errors
        ],
    }

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2)

    log.info(
        "Missing-slices report written to: %s  (%d stack(s) with gaps)",
        report_path, len(missing_report),
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert z-stack image folders to NIfTI (.nii) volumes "
            "for visualization in ITK-SNAP."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "stack_dir",
        nargs="?",
        metavar="STACK_DIR",
        help="Path to a single stack folder (containing _zXX_chYY.jpg files).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Discover and process every stack under "
            "'Cell counting/In vitro experiment'."
        ),
    )
    parser.add_argument(
        "--xy-pixel-um",
        type=float,
        default=1.5, #actual project step, can be redefined in the future!
        metavar="UM",
        help="In-plane pixel size in µm (default: 1.0, placeholder).",
    )
    parser.add_argument(
        "--z-step-um",
        type=float,
        default=5.0, #actual project step, can be redefined in the future!
        metavar="UM",
        help="Z-step between slices in µm (default: 1.0, placeholder).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.all:
        batch_generate_niftis(
            xy_pixel_um=args.xy_pixel_um,
            z_step_um=args.z_step_um,
        )
    else:
        generate_nifti_and_segmentation(
            stack_dir=args.stack_dir,
            xy_pixel_um=args.xy_pixel_um,
            z_step_um=args.z_step_um,
        )
