"""Importing this package triggers registration of every metric provider."""
from . import shape  # noqa: F401
from . import intensity  # noqa: F401
from . import derived  # noqa: F401
from .registry import MetricGroup, GROUP_ORDER, providers, register

__all__ = ["MetricGroup", "GROUP_ORDER", "providers", "register"]
