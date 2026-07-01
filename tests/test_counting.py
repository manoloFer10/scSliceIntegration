"""End-to-end counting: synthetic TIF series → convert → count_tree → counts CSV.

Also exercises the CLI dispatch (bare-root → metrics back-compat, and the `count`
subcommand).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pd = pytest.importorskip("pandas")

from sc3d.cli import main
from sc3d.core.counting import count_tree
from sc3d.preprocess.leica import SeriesRef, convert_series


def _write_rgb_tif(path: Path, green_slice: np.ndarray) -> None:
    y, x = green_slice.shape
    rgb = np.zeros((y, x, 3), dtype=np.uint8)
    rgb[..., 1] = green_slice
    sitk.WriteImage(sitk.GetImageFromArray(rgb, isVector=True), str(path))


def _make_blob_series(
    series_dir: Path,
    blob_centers: list[tuple[int, int, int]],
    vol_shape: tuple[int, int, int] = (20, 64, 64),
    blob_radius: int = 4,
    intensity: int = 200,
) -> int:
    """Write a synthetic ch00 z-series with K well-separated blobs; return K."""
    series_dir.mkdir(parents=True, exist_ok=True)
    Z, Y, X = vol_shape
    volume = np.zeros(vol_shape, dtype=np.uint8)
    for cz, cy, cx in blob_centers:
        zs, ys, xs = np.ogrid[:Z, :Y, :X]
        volume[(zs - cz) ** 2 + (ys - cy) ** 2 + (xs - cx) ** 2 <= blob_radius ** 2] = intensity
    for z in range(Z):
        _write_rgb_tif(series_dir / f"series_z{z:02d}_ch00.tif", volume[z])
    return len(blob_centers)


def _convert_one_series(data_root: Path, out_root: Path) -> tuple[int, str]:
    """Build a K-blob series, convert it to a .nii.gz tree; return (K, sample)."""
    centers = [(5, 16, 16), (10, 16, 48), (15, 48, 32)]
    K = _make_blob_series(data_root / "Series1", centers)
    ref = SeriesRef(tissue="TST", day="Day_1", series="Series1", series_dir=data_root / "Series1")
    convert_series(ref, out_root, expected_z=20, threshold=100, min_volume_um3=1.0, denoise_radius=0)
    return K, "TST/Day_1/Series1"


def test_count_tree_matches_k(tmp_path: Path) -> None:
    out_root = tmp_path / "niftis"
    K, sample = _convert_one_series(tmp_path / "data", out_root)

    csv_path = count_tree(out_root)
    assert csv_path.name == "all_samples_counts.csv"

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["sample", "n_objects"]
    row = df[df["sample"] == sample]
    assert len(row) == 1
    assert int(row["n_objects"].iloc[0]) == K


def test_cli_count_subcommand(tmp_path: Path) -> None:
    out_root = tmp_path / "niftis"
    K, sample = _convert_one_series(tmp_path / "data", out_root)

    rc = main(["count", str(out_root), "--out", str(tmp_path)])
    assert rc == 0

    df = pd.read_csv(tmp_path / "all_samples_counts.csv")
    assert int(df[df["sample"] == sample]["n_objects"].iloc[0]) == K


def test_cli_bare_root_dispatches_to_metrics(tmp_path: Path) -> None:
    """`python -m sc3d <root>` (no subcommand) still runs the metrics path."""
    root = tmp_path / "empty_tree"
    root.mkdir()
    rc = main([str(root)])  # no stacks → header-only aggregate, no error
    assert rc == 0
    assert (root / "all_samples_metrics.csv").exists()
