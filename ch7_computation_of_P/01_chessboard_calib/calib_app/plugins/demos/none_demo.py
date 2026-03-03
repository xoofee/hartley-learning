"""Default demo: no special behavior."""
from __future__ import annotations

from ..registry import Demo


class NoneDemo(Demo):
    def id(self) -> str:
        return "none"

    def label(self) -> str:
        return "None"
