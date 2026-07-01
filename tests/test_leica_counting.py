"""Tests for the Leica TIF → NIfTI converter.

Build synthetic RGB TIF z-series with K well-separated blobs, run convert_series,
and verify blob count, spacing, and fluor signal.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from sc3d.preprocess.leica import SeriesRef, convert_series, iter_series, load_channel_volume


def _write_rgb_tif(path: Path, green_slice: np.ndarray) -> None:
    """Write a (Y, X) grayscale array as an RGB TIF with signal in the green channel."""
    y, x = green_slice.shape
    rgb = np.zeros((y, x, 3), dtype=np.uint8)
    rgb[..., 1] = green_slice  # green channel
    img = sitk.GetImageFromArray(rgb, isVector=True)
    sitk.WriteImage(img, str(path))


def _make_blob_volume(
    tmp_path: Path,
    series_name: str,
    blob_centers: list[tuple[int, int, int]],  # (z, y, x) 0-based
    vol_shape: tuple[int, int, int] = (20, 64, 64),  # (Z, Y, X)
    blob_radius: int = 4,
    intensity: int = 200,
) -> tuple[Path, int]:
    """Write a synthetic z-series TIF set with K blobs; return (series_dir, K)."""
    series_dir = tmp_path / series_name
    series_dir.mkdir(parents=True)

    Z, Y, X = vol_shape
    volume = np.zeros(vol_shape, dtype=np.uint8)
    for cz, cy, cx in blob_centers:
        zs, ys, xs = np.ogrid[:Z, :Y, :X]
        mask = (zs - cz) ** 2 + (ys - cy) ** 2 + (xs - cx) ** 2 <= blob_radius ** 2
        volume[mask] = intensity

    for z_idx in range(Z):
        fname = series_dir / f"series_z{z_idx:02d}_ch00.tif"
        _write_rgb_tif(fname, volume[z_idx])

    return series_dir, len(blob_centers)


def test_blob_count_matches_k(tmp_path: Path) -> None:
    """convert_series on K well-separated blobs → segmentation has exactly K components."""
    K = 3
    blob_centers = [(5, 16, 16), (10, 16, 48), (15, 48, 32)]
    series_dir, _ = _make_blob_volume(tmp_path / "data", "Series1", blob_centers)

    ref = SeriesRef(tissue="TST", day="Day_1", series="Series1", series_dir=series_dir)
    out_root = tmp_path / "niftis"
    convert_series(ref, out_root, expected_z=20, threshold=100, min_volume_um3=1.0, bg_radius=5)

    seg_path = out_root / "TST" / "Day_1" / "Series1" / "Series1_segmentation.nii.gz"
    assert seg_path.exists(), "segmentation file not written"

    seg = sitk.ReadImage(str(seg_path))
    cc = sitk.ConnectedComponent(seg)
    n_labels = int(sitk.GetArrayFromImage(cc).max())
    assert n_labels == K, f"expected {K} components, got {n_labels}"


def test_spacing_set_correctly(tmp_path: Path) -> None:
    """Fluor NIfTI must carry the correct physical spacing, not the TIF default."""
    series_dir, _ = _make_blob_volume(tmp_path / "data", "Series1", [(10, 32, 32)])

    ref = SeriesRef(tissue="TST", day="Day_1", series="Series1", series_dir=series_dir)
    out_root = tmp_path / "niftis"
    convert_series(ref, out_root, expected_z=20, threshold=100, min_volume_um3=1.0, bg_radius=5)

    fluor_path = out_root / "TST" / "Day_1" / "Series1" / "Series1_fluor.nii.gz"
    fluor = sitk.ReadImage(str(fluor_path))
    sx, sy, sz = fluor.GetSpacing()
    assert sx == pytest.approx(1.1353, abs=1e-3)
    assert sy == pytest.approx(1.1353, abs=1e-3)
    assert sz == pytest.approx(5.0, abs=1e-3)


def test_fluor_max_positive(tmp_path: Path) -> None:
    """Fluor NIfTI must contain positive values (green channel was non-zero)."""
    series_dir, _ = _make_blob_volume(tmp_path / "data", "Series1", [(10, 32, 32)])

    ref = SeriesRef(tissue="TST", day="Day_1", series="Series1", series_dir=series_dir)
    out_root = tmp_path / "niftis"
    convert_series(ref, out_root, expected_z=20, threshold=100, min_volume_um3=1.0, bg_radius=5)

    fluor_path = out_root / "TST" / "Day_1" / "Series1" / "Series1_fluor.nii.gz"
    fluor = sitk.ReadImage(str(fluor_path))
    arr = sitk.GetArrayFromImage(fluor)
    assert arr.max() > 0


def test_iter_series_discovers_folders(tmp_path: Path) -> None:
    """iter_series must yield folders that have ch00 TIFs and skip others."""
    # Create a valid series
    valid = tmp_path / "Analysis_COR (Cortex)" / "Analysis_Day 1" / "Series1"
    valid.mkdir(parents=True)
    (valid / "img_z00_ch00.tif").write_bytes(b"")  # touch a fake TIF

    # A folder without ch00 TIFs should be skipped
    empty = tmp_path / "Analysis_COR (Cortex)" / "Analysis_Day 1" / "Empty"
    empty.mkdir()

    refs = list(iter_series(tmp_path))
    assert len(refs) == 1
    r = refs[0]
    assert r.tissue == "COR"
    assert r.day == "Day_1"
    assert r.series == "Series1"
    assert r.series_dir == valid


def test_load_channel_volume_extracts_green(tmp_path: Path) -> None:
    """load_channel_volume must extract the green (index 1) RGB channel."""
    series_dir = tmp_path / "Series1"
    series_dir.mkdir()

    z_shape = (8, 16, 16)
    green_val = 137
    volume = np.zeros(z_shape, dtype=np.uint8)
    volume[4, 8, 8] = green_val

    for z in range(z_shape[0]):
        _write_rgb_tif(series_dir / f"img_z{z:02d}_ch00.tif", volume[z])

    img = load_channel_volume(series_dir, channel="ch00", expected_z=z_shape[0])
    arr = sitk.GetArrayFromImage(img)
    assert arr.max() == green_val
    assert arr.shape == z_shape


def test_z_mismatch_raises(tmp_path: Path) -> None:
    """load_channel_volume must raise ValueError when slice count != expected_z."""
    series_dir = tmp_path / "Series1"
    series_dir.mkdir()
    for z in range(5):
        _write_rgb_tif(series_dir / f"img_z{z:02d}_ch00.tif", np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected 10"):
        load_channel_volume(series_dir, channel="ch00", expected_z=10)
