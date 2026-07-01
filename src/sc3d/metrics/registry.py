"""Metric registration. Providers self-register via @register on import."""
from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable, TypeVar

from .base import MetricProvider


class MetricGroup(str, Enum):
    SHAPE = "shape"
    INTENSITY = "intensity"
    DERIVED = "derived"


# Fixed order: derived metrics depend on shape + intensity output.
GROUP_ORDER: tuple[MetricGroup, ...] = (
    MetricGroup.SHAPE,
    MetricGroup.INTENSITY,
    MetricGroup.DERIVED,
)


_REGISTRY: dict[MetricGroup, list[MetricProvider]] = defaultdict(list)

P = TypeVar("P", bound=MetricProvider)


def register(group: MetricGroup) -> Callable[[type[P]], type[P]]:
    def decorator(cls: type[P]) -> type[P]:
        _REGISTRY[group].append(cls())
        return cls
    return decorator


def providers(group: MetricGroup) -> list[MetricProvider]:
    return list(_REGISTRY[group])
