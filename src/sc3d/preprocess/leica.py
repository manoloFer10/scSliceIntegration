"""TIF → NIfTI converter for Leica confocal z-stacks.

Each Series folder under data/Analysis_<TISSUE> (...)/Analysis_Day {N}/ contains
100 z-slices × 2 channels as RGB 8-bit TIFs.  This module:
  1. Discovers series folders (iter_series).
  2. Loads the GFP channel (ch00, green = RGB index 1) into a 3-D SimpleITK image
     with correct anisotropic spacing (load_channel_volume + resolve_spacing).
  3. Segments foreground blobs (segment_fluorescence from core.segment).
  4. Writes <series>_fluor.nii.gz and <series>_segmentation.nii.gz (convert_series).
  5. Loops over all series with per-series error isolation (convert).

Output tree mirrors the NiftiStackLoader convention so the existing metrics pipeline
runs unchanged on the output root.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree

import numpy as np
import SimpleITK as sitk

from sc3d.core.segment import segment_fluorescence

logger = logging.getLogger(__name__)

# Default spacing confirmed from Leica metadata (DimensionDescription Voxel values).
# The TIF-embedded 0.35278 µm is a bogus 1/72-inch screen default — never use it.
_DEFAULT_SPACING = (1.1353, 1.1353, 5.0)  # (sx, sy, sz) in µm


@dataclass(frozen=True)
class SeriesRef:
    tissue: str       # e.g. "COR" or "WER"
    day: str          # e.g. "Day_1"
    series: str       # verbatim folder name, e.g. "Series1" or "Series002"
    series_dir: Path


def iter_series(data_root: Path) -> Iterator[SeriesRef]:
    """Yield one SeriesRef per Series folder that contains ch00 TIFs.

    Folder structure expected:
        data_root/
          Analysis_<TISSUE> (...)/
            Analysis_Day {N}/
              <Series>/
                *_ch00.tif   (required; absence → folder skipped)
    """
    tissue_re = re.compile(r"Analysis_(\w+)\s")
    day_re = re.compile(r"Analysis_Day\s+(\d+)")

    for tissue_dir in sorted(data_root.iterdir()):
        if not tissue_dir.is_dir():
            continue
        m = tissue_re.search(tissue_dir.name)
        if not m:
            continue
        tissue = m.group(1)

        for day_dir in sorted(tissue_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            m = day_re.search(day_dir.name)
            if not m:
                continue
            day = f"Day_{m.group(1)}"

            for series_dir in sorted(day_dir.iterdir()):
                if not series_dir.is_dir():
                    continue
                if not list(series_dir.glob("*_ch00.tif")):
                    continue
                yield SeriesRef(
                    tissue=tissue,
                    day=day,
                    series=series_dir.name,
                    series_dir=series_dir,
                )


def resolve_spacing(series_dir: Path) -> tuple[float, float, float]:
    """Return (sx, sy, sz) in µm from MetaData XML, falling back to defaults.

    Looks for MetaData/*_Properties.xml → DimensionDescription[@DimID] Voxel attr.
    Comma-formatted floats (e.g. "1,136") are normalised to decimal dots.
    Falls back to _DEFAULT_SPACING on any parse error or missing file.
    """
    xml_files = list((series_dir / "MetaData").glob("*_Properties.xml")) if (series_dir / "MetaData").is_dir() else []
    if not xml_files:
        return _DEFAULT_SPACING

    try:
        tree = ElementTree.parse(xml_files[0])
        root = tree.getroot()
        voxels: dict[str, float] = {}
        for dim in root.iter("DimensionDescription"):
            dim_id = dim.get("DimID", "")
            raw = dim.get("Voxel", "")
            if dim_id in ("X", "Y", "Z") and raw:
                voxels[dim_id] = float(raw.replace(",", "."))
        if "X" in voxels and "Y" in voxels and "Z" in voxels:
            return (voxels["X"], voxels["Y"], voxels["Z"])
    except Exception as exc:
        logger.warning("Could not parse spacing from %s: %s — using defaults", xml_files[0], exc)

    return _DEFAULT_SPACING


def load_channel_volume(
    series_dir: Path,
    channel: str = "ch00",
    expected_z: int | None = 100,
) -> sitk.Image:
    """Load all z-slices for *channel* into a (Z, Y, X) uint8 SimpleITK image.

    TIFs are RGB 8-bit; green (GFP) signal lives in RGB index 1.
    Slices are sorted by the integer extracted from _z{NN}_{channel}.tif.
    Spacing is set by resolve_spacing (correct physical µm, not TIF default).

    Raises ValueError if expected_z is set and the found count does not match.
    """
    pattern = re.compile(rf"_z(\d+)_{re.escape(channel)}\.tif$")
    tif_files: list[tuple[int, Path]] = []
    for p in series_dir.iterdir():
        m = pattern.search(p.name)
        if m:
            tif_files.append((int(m.group(1)), p))

    if not tif_files:
        raise FileNotFoundError(f"No *_{channel}.tif slices found in {series_dir}")

    tif_files.sort(key=lambda t: t[0])

    if expected_z is not None and len(tif_files) != expected_z:
        raise ValueError(
            f"Expected {expected_z} z-slices for {channel} in {series_dir.name}, "
            f"found {len(tif_files)}"
        )

    slices: list[np.ndarray] = []
    for _, path in tif_files:
        rgb = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # (Y, X, 3)
        slices.append(rgb[..., 1])  # green channel

    volume = np.stack(slices, axis=0).astype(np.uint8)  # (Z, Y, X)
    img = sitk.GetImageFromArray(volume)
    sx, sy, sz = resolve_spacing(series_dir)
    img.SetSpacing((sx, sy, sz))
    return img


def convert_series(
    ref: SeriesRef,
    out_root: Path,
    channel: str = "ch00",
    raw_channel: str = "ch01",
    expected_z: int | None = 100,
    **seg_kwargs,
) -> None:
    """Convert one series: load → segment → write fluor + segmentation + raw NIfTIs.

    Output directory: out_root/<tissue>/<day>/<series>/
    Output files:
        <series>_fluor.nii.gz          — GFP intensity (uint8, green channel of ch00)
        <series>_segmentation.nii.gz   — binary foreground mask (uint8)
        <series>_raw.nii.gz            — brightfield (uint8, green channel of ch01)
    The raw file is omitted with a warning if raw_channel TIFs are absent.
    """
    fluor = load_channel_volume(ref.series_dir, channel=channel, expected_z=expected_z)
    seg = segment_fluorescence(fluor, **seg_kwargs)

    out_dir = out_root / ref.tissue / ref.day / ref.series
    out_dir.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(fluor, str(out_dir / f"{ref.series}_fluor.nii.gz"))
    sitk.WriteImage(seg, str(out_dir / f"{ref.series}_segmentation.nii.gz"))

    try:
        raw = load_channel_volume(ref.series_dir, channel=raw_channel, expected_z=expected_z)
        sitk.WriteImage(raw, str(out_dir / f"{ref.series}_raw.nii.gz"))
    except FileNotFoundError:
        logger.warning("No %s TIFs in %s — skipping raw volume", raw_channel, ref.series_dir.name)

    logger.info("Written %s/%s/%s", ref.tissue, ref.day, ref.series)


def convert(
    data_root: Path,
    out_root: Path,
    channel: str = "ch00",
    **seg_kwargs,
) -> None:
    """Convert all series under data_root, skipping failures with a logged warning."""
    for ref in iter_series(data_root):
        try:
            convert_series(ref, out_root, channel=channel, **seg_kwargs)
        except Exception as exc:
            logger.warning(
                "Skipping %s/%s/%s: %s",
                ref.tissue, ref.day, ref.series, exc,
            )
