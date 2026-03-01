"""
Built-in demos: exclusive modes (P row planes, Backproject, Angulometer).

Each demo is independent; when its button is off, on_deactivated() releases all state.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QButtonGroup, QPushButton

from .registry import Demo, register_demo
from .. import geometry
from .. import distortion


class NoneDemo(Demo):
    """No demo active."""

    def id(self) -> str:
        return "none"

    def label(self) -> str:
        return "None"

    def on_deactivated(self) -> None:
        pass


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


class AngulometerDemo(Demo):
    """
    Angle between two points by K: user drags two image points.
    Show angle dynamically on image plot and both rays in 3D.
    When button is off, release all related state.
    """

    def __init__(self):
        self._points_uv: list[tuple[float, float]] = []  # up to 2 points (ideal coords)
        self._dragging_index: int | None = None  # 0 or 1 when dragging

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
        w = context.get("image_width_px", 0) or 1
        h = context.get("image_height_px", 0) or 1
        # Start with two points: e.g. center and offset
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
        # Angle text
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


def build_demos_button_group(parent) -> tuple[QGroupBox, QButtonGroup, dict[str, QPushButton]]:
    """Create a Demos group with exclusive buttons. Returns (group_widget, button_group, id_to_button)."""
    from .registry import get_demos
    group = QGroupBox("Demos")
    layout = QVBoxLayout()
    button_group = QButtonGroup(parent)
    button_group.setExclusive(True)
    id_to_button: dict[str, QPushButton] = {}
    for demo in get_demos():
        btn = QPushButton(demo.label())
        btn.setCheckable(True)
        if demo.id() == "none":
            btn.setChecked(True)
        btn.setProperty("demo_id", demo.id())
        button_group.addButton(btn)
        layout.addWidget(btn)
        id_to_button[demo.id()] = btn
    group.setLayout(layout)
    return group, button_group, id_to_button


def register_builtin_demos() -> None:
    """Register None, P row planes, Backproject, Angulometer demos."""
    register_demo(NoneDemo())
    register_demo(PRowPlanesDemo())
    register_demo(BackprojectDemo())
    register_demo(AngulometerDemo())
