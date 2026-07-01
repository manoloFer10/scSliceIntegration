"""CSV writers for per-stack and aggregated outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schema import CANONICAL_COLUMNS


def _to_dataframe(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    # Add any missing columns as NaN; reorder to canonical.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    extras = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    return df[list(CANONICAL_COLUMNS) + extras]


def write_per_stack(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_dataframe(rows).to_csv(path, index=False)


def write_aggregate(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_dataframe(rows).to_csv(path, index=False)
