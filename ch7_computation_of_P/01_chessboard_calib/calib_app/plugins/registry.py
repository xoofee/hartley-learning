"""
Plugin/demo registry: add new demos without changing core GUI.
Exclusive demos (only one active at a time).
"""
from __future__ import annotations

from typing import Any

import numpy as np

_DEMOS: list["Demo"] = []
_DEMOS_BY_ID: dict[str, "Demo"] = {}


class Demo:
    """
    One demo/mode. Buttons are exclusive; only one demo active.
    When demo is off, on_deactivated() is called to release state.
    """

    def id(self) -> str:
        """Unique id (e.g. 'none', 'realtime_pose')."""
        raise NotImplementedError

    def label(self) -> str:
        """Short label for the demo button."""
        raise NotImplementedError

    def on_activated(self, context: dict) -> None:
        """Called when this demo becomes active."""
        pass

    def on_deactivated(self) -> None:
        """Called when this demo is turned off; release state."""
        pass

    def hide_calibration_pyramids(self) -> bool:
        """If True, 3D plot will not draw calibration camera pyramids (chessboard still drawn)."""
        return False

    def on_frame(self, frame_bgr: np.ndarray, context: dict) -> None:
        """Called when a new camera frame is available (preview on). Demo may set context['state'].realtime_pose."""
        pass


def register_demo(demo: Demo) -> None:
    """Register a demo; order of registration is button order. Idempotent per id."""
    if demo.id() in _DEMOS_BY_ID:
        return
    _DEMOS.append(demo)
    _DEMOS_BY_ID[demo.id()] = demo


def get_demos() -> list[Demo]:
    return list(_DEMOS)


def get_demo_by_id(demo_id: str) -> Demo | None:
    return _DEMOS_BY_ID.get(demo_id)
