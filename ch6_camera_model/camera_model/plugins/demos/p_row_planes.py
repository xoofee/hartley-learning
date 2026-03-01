"""Show the three P row planes in 3D."""
from __future__ import annotations

from ..registry import Demo
from ... import geometry


class PRowPlanesDemo(Demo):
    """Show the three P row planes in 3D. No image events."""

    def id(self) -> str:
        return "p_planes"

    def label(self) -> str:
        return "P row planes"

    def on_activated(self, context: dict) -> None:
        pass

    def on_deactivated(self) -> None:
        pass

    def on_draw_3d(self, ax3d, context: dict) -> None:
        P = context.get("P")
        xlim = context.get("xlim")
        ylim = context.get("ylim")
        zlim = context.get("zlim")
        if P is not None and xlim and ylim and zlim:
            geometry.draw_P_row_planes(ax3d, P, xlim, ylim, zlim)
