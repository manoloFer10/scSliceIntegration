"""Sanity tests on synthetic volumes — analytic ground truth.

Run: `pytest tests/`. Skips automatically if SimpleITK isn't installed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from sc3d import metrics  # registers providers
from sc3d.core.labeling import label_and_filter
from sc3d.core.pipeline import compute_all, run
from sc3d.core.stack import Stack
from sc3d.io.base import StackRef


def _make_sphere_volume(radius: int = 10, shape: tuple[int, int, int] = (40, 40, 40)) -> sitk.Image:
    z, y, x = np.indices(shape)
    cz, cy, cx = (s // 2 for s in shape)
    mask = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= radius ** 2
    arr = mask.astype(np.uint16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def _make_fluor_uniform(shape: tuple[int, int, int]) -> sitk.Image:
    arr = np.full(shape, 100.0, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def test_sphere_metrics_close_to_analytic():
    seg = _make_sphere_volume(radius=10)
    fluor = _make_fluor_uniform((40, 40, 40))
    stack = Stack(sample="synthetic/sphere", segmentation=seg, fluor=fluor, spacing=(1.0, 1.0, 1.0))
    stack, counts = label_and_filter(stack, min_voxels=10)
    rows = compute_all(stack, counts)

    assert len(rows) == 1
    row = rows[0]

    # Sphere of radius 10 voxels: V = 4/3 π r³ ≈ 4188 voxels.
    assert row["n_voxels"] == pytest.approx(4188, rel=0.05)
    assert row["volume_phys"] == pytest.approx(4188.0, rel=0.05)
    assert row["sphericity"] == pytest.approx(1.0, abs=0.15)
    assert row["roundness"] == pytest.approx(1.0, abs=0.15)
    assert row["elongation"] == pytest.approx(1.0, abs=0.05)
    assert row["flatness"] == pytest.approx(1.0, abs=0.05)
    # Uniform fluor → mean == 100, weighted centroid coincides with geometric.
    assert row["fluor_mean"] == pytest.approx(100.0)
    assert row["fluor_cv"] == pytest.approx(0.0, abs=1e-6)
    assert row["centroid_displacement"] == pytest.approx(0.0, abs=1e-3)


def test_min_voxel_filter_drops_noise():
    arr = np.zeros((20, 20, 20), dtype=np.uint16)
    arr[1, 1, 1] = 1                  # 1-voxel speck → must be dropped
    arr[5:15, 5:15, 5:15] = 1          # 1000-voxel cube → kept
    seg = sitk.GetImageFromArray(arr)
    seg.SetSpacing((1.0, 1.0, 1.0))
    stack = Stack(sample="synthetic/noise", segmentation=seg,
                  fluor=_make_fluor_uniform((20, 20, 20)), spacing=(1.0, 1.0, 1.0))
    stack, counts = label_and_filter(stack, min_voxels=10)
    assert list(counts.values()) == [1000]


def test_pluggable_loader_inmemory(tmp_path: Path):
    """Prove the I/O abstraction: a fake loader plugs in unchanged."""
    seg = _make_sphere_volume(radius=8, shape=(30, 30, 30))
    fluor = _make_fluor_uniform((30, 30, 30))

    class InMemoryLoader:
        def __init__(self, stack_dir: Path):
            self._stack_dir = stack_dir

        def discover(self, root: Path) -> Iterator[StackRef]:
            yield StackRef(
                sample="mem/one",
                stack_dir=self._stack_dir,
                seg_path=self._stack_dir / "x_segmentation.nii",  # never read
                fluor_path=self._stack_dir / "x_fluor.nii",
            )

        def load(self, ref: StackRef) -> Stack:
            return Stack(sample=ref.sample, segmentation=seg, fluor=fluor,
                         spacing=(1.0, 1.0, 1.0))

    out = run(InMemoryLoader(tmp_path), tmp_path)
    assert out.exists()
    text = out.read_text()
    assert "sample,label,n_voxels" in text
    assert "mem/one" in text


def test_provider_registry_has_all_groups():
    for g in metrics.GROUP_ORDER:
        assert metrics.providers(g), f"no providers registered for {g}"
