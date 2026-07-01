"""In-memory stack representation handed to metric providers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import SimpleITK as sitk


@dataclass
class Stack:
    sample: str
    segmentation: sitk.Image
    fluor: Optional[sitk.Image]
    spacing: tuple[float, float, float]
