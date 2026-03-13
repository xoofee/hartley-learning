"""Drag one image point; show backprojected ray in 3D."""
from __future__ import annotations

from ..registry import Demo
from ... import geometry
from ... import distortion


class BackprojectDemo(Demo):
    """Drag one image point; show backprojected ray in 3D."""

    def __init__(self):
        self._point_uv: tuple[float, float] | None = None
        self._dragging = False

    def id(self) -> str:
        return "backproject"

    def label(self) -> str:
        return "Backproject (1 pt)"

    def needs_image_events(self) -> bool:
        return True

    def on_activated(self, context: dict) -> None:
        state = context.get("state")
        if state is not None and self._point_uv is None:
            K = state.get_K()
            cx, cy = float(K[0, 2]), float(K[1, 2])
            self._point_uv = (cx, cy)

    def on_deactivated(self) -> None:
        self._point_uv = None
        self._dragging = False

    def set_point_uv(self, u: float, v: float) -> None:
        self._point_uv = (u, v)

    def set_dragging(self, value: bool) -> None:
        self._dragging = value

    def is_dragging(self) -> bool:
        return self._dragging

    def on_image_button_press(self, event, context: dict) -> None:
        if event.inaxes is None:
            return
        self._dragging = True

    def on_image_motion(self, event, context: dict) -> None:
        pass  # App sets point via set_point_uv after undistort

    def on_image_button_release(self, event, context: dict) -> None:
        self._dragging = False

    def on_draw_image(self, ax_img, context: dict) -> None:
        if self._point_uv is None:
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

    def on_draw_3d(self, ax3d, context: dict) -> None:
        if self._point_uv is None:
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
