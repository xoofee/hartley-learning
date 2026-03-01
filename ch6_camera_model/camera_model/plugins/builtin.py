"""
Built-in features: P row planes visibility, backproject (drag image point).

These are implemented as plugins so the main app stays open for new features.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox

from .registry import Feature, register_feature
from .. import geometry
from .. import distortion


class PRowPlanesFeature(Feature):
    """Show/hide the three P row planes in 3D view."""

    def __init__(self):
        self._checkbox: QCheckBox | None = None

    def checkbox_widget(self, parent):
        self._checkbox = QCheckBox("Show P row planes (3D)")
        self._checkbox.setChecked(False)
        return self._checkbox

    def is_checked(self) -> bool:
        return self._checkbox.isChecked() if self._checkbox else False

    def on_draw_3d(self, ax3d, context):
        if not self.is_checked():
            return
        P = context.get("P")
        xlim = context.get("xlim")
        ylim = context.get("ylim")
        zlim = context.get("zlim")
        if P is not None and xlim and ylim and zlim:
            geometry.draw_P_row_planes(ax3d, P, xlim, ylim, zlim)


class BackprojectFeature(Feature):
    """Drag image point and show backprojected ray in 3D."""

    def __init__(self):
        self._checkbox: QCheckBox | None = None
        self._point_uv: tuple[float, float] | None = None
        self._dragging = False
        self._cids: list = []

    def checkbox_widget(self, parent):
        self._checkbox = QCheckBox("Backproject (drag image point)")
        self._checkbox.setChecked(False)
        return self._checkbox

    def on_toggled(self, checked: bool):
        self._dragging = False

    def set_point_from_context(self, context: dict):
        if context is None:
            return
        state = context.get("state")
        if state is not None and self._point_uv is None:
            K = state.get_K()
            cx, cy = float(K[0, 2]), float(K[1, 2])
            self._point_uv = (cx, cy)

    def is_checked(self) -> bool:
        return self._checkbox.isChecked() if self._checkbox else False

    def get_point_uv(self):
        return self._point_uv

    def set_point_uv(self, u: float, v: float):
        self._point_uv = (u, v)

    def set_dragging(self, value: bool):
        self._dragging = value

    def is_dragging(self) -> bool:
        return self._dragging

    def on_draw_image(self, ax_img, context):
        if not self.is_checked() or self._point_uv is None:
            return
        state = context.get("state")
        if state is None:
            return
        u, v = self._point_uv
        dist = state.get_distortion()
        K = state.get_K()
        if distortion.distortion_params_nonzero(*dist):
            u_d, v_d = distortion.apply_distortion(u, v, K, *dist)
        else:
            u_d, v_d = u, v
        ax_img.scatter(u_d, v_d, c="orange", s=100, zorder=10, edgecolors="white", linewidths=2)

    def on_draw_3d(self, ax3d, context):
        if not self.is_checked() or self._point_uv is None:
            return
        state = context.get("state")
        if state is None:
            return
        u, v = self._point_uv
        R_cw, t = state.get_R_and_t()
        K = state.get_K()
        C, d = geometry.backproject_image_point_to_ray(u, v, K, R_cw, t)
        ray_scale = 8.0
        pt_far = C + ray_scale * d
        ax3d.plot(
            [C[0], pt_far[0]],
            [C[1], pt_far[1]],
            [C[2], pt_far[2]],
            "o-",
            color="orange",
            linewidth=2,
            markersize=6,
        )


def register_builtin_features():
    """Register P row planes and backproject with the plugin registry."""
    register_feature("show_P_planes", PRowPlanesFeature())
    register_feature("backproject", BackprojectFeature())
