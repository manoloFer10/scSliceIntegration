"""Loader abstraction. Swap this layer to support a new input format."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from ..core.stack import Stack


@dataclass(frozen=True)
class StackRef:
    """Lightweight handle to a stack on disk; loading is deferred."""
    sample: str          # e.g. "Day_1/Alginate/well 1_Alginate_z stack 1"
    stack_dir: Path      # directory that holds the per-stack CSV output
    seg_path: Path
    fluor_path: Path


class StackLoader(Protocol):
    def discover(self, root: Path) -> Iterator[StackRef]: ...
    def load(self, ref: StackRef) -> Stack: ...
