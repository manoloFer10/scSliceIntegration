"""MetricProvider protocol — every metric module implements this."""
from __future__ import annotations

from typing import Protocol

from ..core.stack import Stack

# Per-object row: {label: {column: value}}.
RowMap = dict[int, dict[str, float]]


class MetricProvider(Protocol):
    name: str
    columns: tuple[str, ...]

    def compute(self, stack: Stack, prior: RowMap) -> RowMap: ...
