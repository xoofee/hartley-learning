"""Affine (telecentric) camera demo: projection without perspective divide."""
from __future__ import annotations

from ..registry import Demo


class AffineTelecentricDemo(Demo):
    """
    When active, the image is rendered with an affine (telecentric) camera model:
    (u, v) = (row0·X, row1·X) with no division by depth. Parallel lines in 3D
    remain parallel in the image; no vanishing points.
    """

    def id(self) -> str:
        return "affine"

    def label(self) -> str:
        return "Affine (telecentric)"

    def on_activated(self, context: dict) -> None:
        pass

    def on_deactivated(self) -> None:
        pass

    def on_draw_3d(self, ax3d, context: dict) -> None:
        pass

    def on_draw_image(self, ax_img, context: dict) -> None:
        # Optional: draw a label that we're in affine mode
        ax_img.annotate(
            "Affine (telecentric)",
            (0.02, 0.98),
            xycoords="axes fraction",
            fontsize=9,
            va="top",
            color="gray",
        )
