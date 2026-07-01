"""Discovery must match both `.nii` (original tree) and `.nii.gz` (Leica output)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from sc3d.io.discovery import walk_segmentations


def _write_nii(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))  # format inferred from suffix


def _stack_dir(root: Path, sample: str) -> Path:
    d = root / sample
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_discovers_nii_gz_and_pairs_fluor(tmp_path: Path) -> None:
    d = _stack_dir(tmp_path, "COR/Day_1/Series1")
    _write_nii(d / "Series1_segmentation.nii.gz", np.zeros((4, 4, 4), np.uint8))
    _write_nii(d / "Series1_fluor.nii.gz", np.zeros((4, 4, 4), np.uint8))

    refs = list(walk_segmentations(tmp_path))
    assert len(refs) == 1
    r = refs[0]
    assert r.sample == "COR/Day_1/Series1"
    assert r.seg_path.name == "Series1_segmentation.nii.gz"
    assert r.fluor_path.name == "Series1_fluor.nii.gz"
    assert r.fluor_path.exists()


def test_discovers_plain_nii_identically(tmp_path: Path) -> None:
    """Regression: the uncompressed `.nii` tree is discovered as before."""
    d = _stack_dir(tmp_path, "Day_1/Alginate/well 1")
    _write_nii(d / "well 1_segmentation.nii", np.zeros((4, 4, 4), np.uint8))
    _write_nii(d / "well 1_fluor.nii", np.zeros((4, 4, 4), np.uint8))

    refs = list(walk_segmentations(tmp_path))
    assert len(refs) == 1
    r = refs[0]
    assert r.sample == "Day_1/Alginate/well 1"
    assert r.seg_path.name == "well 1_segmentation.nii"
    assert r.fluor_path.name == "well 1_fluor.nii"
    assert r.fluor_path.exists()


def test_missing_fluor_is_tolerated(tmp_path: Path) -> None:
    d = _stack_dir(tmp_path, "COR/Day_1/Series1")
    _write_nii(d / "Series1_segmentation.nii.gz", np.zeros((4, 4, 4), np.uint8))

    refs = list(walk_segmentations(tmp_path))
    assert len(refs) == 1
    r = refs[0]
    # fluor_path defaults to the segmentation's own extension and does not exist.
    assert r.fluor_path.name == "Series1_fluor.nii.gz"
    assert not r.fluor_path.exists()


def test_both_extensions_same_stem_yields_once(tmp_path: Path) -> None:
    d = _stack_dir(tmp_path, "COR/Day_1/Series1")
    _write_nii(d / "Series1_segmentation.nii", np.zeros((4, 4, 4), np.uint8))
    _write_nii(d / "Series1_segmentation.nii.gz", np.zeros((4, 4, 4), np.uint8))

    refs = list(walk_segmentations(tmp_path))
    assert len(refs) == 1  # deduped on (stack_dir, stem)
