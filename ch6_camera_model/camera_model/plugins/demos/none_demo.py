"""No demo active."""
from __future__ import annotations

from ..registry import Demo


class NoneDemo(Demo):
    """No demo active."""

    def id(self) -> str:
        return "none"

    def label(self) -> str:
        return "None"

    def on_deactivated(self) -> None:
        pass
