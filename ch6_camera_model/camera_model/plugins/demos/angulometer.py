"""
Angle between two points by K: user drags two image points.
Show angle on image and both rays in 3D.
"""
from __future__ import annotations

from ..registry import Demo
from ... import geometry
from ... import distortion


class AngulometerDemo(Demo):
    """
    Angle between two points by K: user drags two image points.
    Show angle dynamically on image plot and both rays in 3D.
    When button is off, release all related state.
    """

    def __init__(self):
        self._points_uv: list[tuple[float, float]] = []
        self._dragging_index: int | None = None

    def id(self) -> str:
        return "angulometer"

    def label(self) -> str:
        return "Angulometer (2 pts)"

    def needs_image_events(self) -> bool:
        return True

    def on_activated(self, context: dict) -> None:
        state = context.get("state")
        if state is None:
            return
        K = state.get_K()
        cx, cy = float(K[0, 2]), float(K[1, 2])
        self._points_uv = [(cx - 30, cy), (cx + 30, cy)]

    def on_deactivated(self) -> None:
        self._points_uv = []
        self._dragging_index = None

    def set_point_uv(self, index: int, u: float, v: float) -> None:
        while len(self._points_uv) <= index:
            self._points_uv.append((0.0, 0.0))
        self._points_uv[index] = (u, v)

    def set_dragging_index(self, index: int | None) -> None:
        self._dragging_index = index

    def get_dragging_index(self) -> int | None:
        return self._dragging_index

    def hit_test(self, u_d: float, v_d: float, context: dict) -> int | None:
        """Return 0 or 1 if (u_d, v_d) is near point 0 or 1 (in distorted coords), else None."""
        state = context.get("state")
        if state is None or len(self._points_uv) < 2:
            return None
        K = state.get_K()
        dist = state.get_distortion()
        margin = 15.0
        for i, (u, v) in enumerate(self._points_uv):
            if distortion.distortion_params_nonzero(*dist):
                ud, vd = distortion.apply_distortion(u, v, K, *dist)
            else:
                ud, vd = u, v
            if abs(ud - u_d) <= margin and abs(vd - v_d) <= margin:
                return i
        return None

    def on_image_button_press(self, event, context: dict) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        idx = self.hit_test(float(event.xdata), float(event.ydata), context)
        self._dragging_index = idx

    def on_image_motion(self, event, context: dict) -> None:
        pass  # App updates point via set_point_uv(index, u, v)

    def on_image_button_release(self, event, context: dict) -> None:
        self._dragging_index = None

    def on_draw_image(self, ax_img, context: dict) -> None:
        state = context.get("state")
        if state is None or len(self._points_uv) < 2:
            return
        K = state.get_K()
        dist = state.get_distortion()
        colors = ["lime", "cyan"]
        for i, (u, v) in enumerate(self._points_uv):
            if distortion.distortion_params_nonzero(*dist):
                u_d, v_d = distortion.apply_distortion(u, v, K, *dist)
            else:
                u_d, v_d = u, v
            ax_img.scatter(u_d, v_d, c=colors[i], s=100, zorder=10, edgecolors="white", linewidths=2)
        angle_deg = self._angle_deg(context)
        if angle_deg is not None:
            mid_u = (self._points_uv[0][0] + self._points_uv[1][0]) / 2
            mid_v = (self._points_uv[0][1] + self._points_uv[1][1]) / 2
            if distortion.distortion_params_nonzero(*dist):
                mid_u, mid_v = distortion.apply_distortion(mid_u, mid_v, K, *dist)
            ax_img.text(
                mid_u, mid_v, f"  {angle_deg:.2f}°",
                fontsize=12, color="yellow", fontweight="bold",
                verticalalignment="center", zorder=11,
            )

    def _angle_deg(self, context: dict) -> float | None:
        if len(self._points_uv) < 2:
            return None
        state = context.get("state")
        if state is None:
            return None
        R_cw, t = state.get_R_and_t()
        K = state.get_K()
        C1, d1 = geometry.backproject_image_point_to_ray(
            self._points_uv[0][0], self._points_uv[0][1], K, R_cw, t
        )
        C2, d2 = geometry.backproject_image_point_to_ray(
            self._points_uv[1][0], self._points_uv[1][1], K, R_cw, t
        )
        return geometry.angle_between_ray_directions_deg(d1, d2)

    def on_draw_3d(self, ax3d, context: dict) -> None:
        if len(self._points_uv) < 2:
            return
        state = context.get("state")
        if state is None:
            return
        R_cw, t = state.get_R_and_t()
        K = state.get_K()
        ray_scale = 8.0
        colors = ["lime", "cyan"]
        for i, (u, v) in enumerate(self._points_uv):
            C, d = geometry.backproject_image_point_to_ray(u, v, K, R_cw, t)
            pt_far = C + ray_scale * d
            ax3d.plot(
                [C[0], pt_far[0]],
                [C[1], pt_far[1]],
                [C[2], pt_far[2]],
                "o-",
                color=colors[i],
                linewidth=2,
                markersize=6,
            )
