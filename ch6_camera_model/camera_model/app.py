"""
Main GUI application: imaging simulation window (P = K [R|t]).

Composes scene, state, widgets, rendering, and plugins. Does not modify
ch6_camera_model/01_imaging_simulation.py (original remains unchanged).
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtCore import Qt

from . import scene
from . import state as state_module
from . import geometry
from . import rendering
from . import distortion
from .logging_ui import set_log_sink
from .widgets import (
    MatrixDisplayWidget,
    MatrixEditWidget,
    CameraParamsWidget,
    DistortionParamsWidget,
    RotationParamsWidget,
    CameraCenterWidget,
    LogOutputWidget,
    ConsoleWidget,
)
from .plugins.registry import get_demo_by_id
from .plugins.demos import register_builtin_demos, build_demos_button_group

# Register built-in demos (exclusive: None, P row planes, Backproject, Angulometer).
register_builtin_demos()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.square_pts = scene.get_scene_square(1.0, 0.0)
        self.triangle_pts = scene.get_scene_triangle(0.8, 1.0, 1.5)
        self.rectangle_pts = scene.get_scene_rectangle(0.4, 0.4, 1.5, y_center=0.8, z_center=0.4)
        self.state = state_module.CameraState()
        self._current_demo_id: str = "none"
        self._demo_cids: list[int] = []
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Camera Geometry")
        self.setGeometry(80, 80, 1400, 700)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        self.fig = plt.figure(figsize=(8, 5))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_img = self.fig.add_subplot(122)
        self._set_3d_axes_limits_once()
        self.canvas = FigureCanvas(self.fig)

        # Left: top = plot, bottom = log (left) + console (right)
        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(self.canvas)
        bottom_split = QSplitter(Qt.Horizontal)
        self.log_widget = LogOutputWidget(title="Log", show_clear_button=True)
        set_log_sink(self.log_widget)
        self.console_widget = ConsoleWidget(
            namespace_getter=self._get_console_namespace,
            title="Console",
        )
        bottom_split.addWidget(self.log_widget)
        bottom_split.addWidget(self.console_widget)
        bottom_split.setSizes([220, 280])
        left_split.addWidget(bottom_split)
        left_split.setStretchFactor(0, 1)
        left_split.setStretchFactor(1, 1)
        left_split.setSizes([350, 350])
        main_layout.addWidget(left_split, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(450)
        scroll_widget = QWidget()
        right_layout = QVBoxLayout()

        self.params_widget = CameraParamsWidget(self.state)
        for spin in (
            self.params_widget.spin_f,
            self.params_widget.spin_wphys,
            self.params_widget.spin_hphys,
            self.params_widget.spin_wpix,
            self.params_widget.spin_hpix,
        ):
            spin.valueChanged.connect(self._on_params_changed)
        self.distortion_widget = DistortionParamsWidget(self.state)
        for spin in (
            self.distortion_widget.spin_dist_k1,
            self.distortion_widget.spin_dist_k2,
            self.distortion_widget.spin_dist_k3,
            self.distortion_widget.spin_dist_p1,
            self.distortion_widget.spin_dist_p2,
        ):
            spin.valueChanged.connect(self._on_distortion_changed)
        camera_and_lens_row = QWidget()
        camera_and_lens_layout = QHBoxLayout()
        camera_and_lens_layout.setContentsMargins(0, 0, 0, 0)
        camera_and_lens_layout.addWidget(self.params_widget)
        camera_and_lens_layout.addWidget(self.distortion_widget)
        camera_and_lens_row.setLayout(camera_and_lens_layout)
        right_layout.addWidget(camera_and_lens_row)

        self.edit_P = MatrixEditWidget("P (3×4) editable", 3, 4)
        self.edit_P.matrix_changed.connect(self._on_P_changed)
        right_layout.addWidget(self.edit_P)
        self.edit_K = MatrixEditWidget("K intrinsic (3×3) editable", 3, 3)
        self.edit_K.matrix_changed.connect(self._on_K_changed)
        right_layout.addWidget(self.edit_K)
        self.display_R = MatrixDisplayWidget("R", 3, 3)
        right_layout.addWidget(self.display_R)

        pose_row = QWidget()
        pose_row_layout = QHBoxLayout()
        pose_row_layout.setContentsMargins(0, 0, 0, 0)
        self.rotation_widget = RotationParamsWidget(self.state)
        for spin in (
            self.rotation_widget.spin_pitch,
            self.rotation_widget.spin_yaw,
            self.rotation_widget.spin_roll,
        ):
            spin.valueChanged.connect(self._on_pose_changed)
        pose_row_layout.addWidget(self.rotation_widget)
        self.C_widget = CameraCenterWidget(self.state)
        for spin in (self.C_widget.spin_Cx, self.C_widget.spin_Cy, self.C_widget.spin_Cz):
            spin.valueChanged.connect(self._on_pose_changed)
        pose_row_layout.addWidget(self.C_widget)
        self.display_t = MatrixDisplayWidget("t = -R@C", 3, 1)
        pose_row_layout.addWidget(self.display_t)
        pose_row_layout.addStretch()
        pose_row.setLayout(pose_row_layout)
        right_layout.addWidget(pose_row)

        # Demos area: exclusive buttons (only one demo active at a time)
        demos_group, self._demos_button_group, self._demos_buttons = build_demos_button_group(self)
        right_layout.addWidget(demos_group)
        for demo_id, btn in self._demos_buttons.items():
            btn.clicked.connect(lambda checked, did=demo_id: self._on_demo_clicked(did))

        right_layout.addStretch()
        scroll_widget.setLayout(right_layout)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, 0)
        central.setLayout(main_layout)

        self._update_matrix_displays()
        self._draw_all()

    def _get_context(self) -> dict:
        """Build context dict for plugin draw callbacks."""
        return {
            "state": self.state,
            "P": self.state.get_P(),
            "xlim": self.ax3d.get_xlim(),
            "ylim": self.ax3d.get_ylim(),
            "zlim": self.ax3d.get_zlim(),
            "image_width_px": getattr(self, "image_width_px", 0),
            "image_height_px": getattr(self, "image_height_px", 0),
            "square_pts": self.square_pts,
            "triangle_pts": self.triangle_pts,
            "rectangle_pts": self.rectangle_pts,
        }

    def _get_console_namespace(self) -> dict:
        """Namespace for the interactive console: P, K, R, t, state, axes, shapes, redraw."""
        def redraw() -> None:
            self.params_widget.sync_from_state()
            self.rotation_widget.sync_from_state()
            self.C_widget.sync_from_state()
            self._update_matrix_displays()
            self._draw_all()

        return {
            "state": self.state,
            "P": self.state.get_P(),
            "K": self.state.get_K(),
            "R": self.state.get_R_and_t()[0],
            "t": self.state.get_R_and_t()[1],
            "fig": self.fig,
            "ax3d": self.ax3d,
            "ax_img": self.ax_img,
            "square_pts": self.square_pts,
            "triangle_pts": self.triangle_pts,
            "rectangle_pts": self.rectangle_pts,
            "redraw": redraw,
            "np": np,
        }

    def _on_demo_clicked(self, demo_id: str) -> None:
        prev_id = self._current_demo_id
        if prev_id == demo_id:
            return
        prev = get_demo_by_id(prev_id)
        if prev is not None:
            prev.on_deactivated()
        self._current_demo_id = demo_id
        current = get_demo_by_id(demo_id)
        if current is not None:
            current.on_activated(self._get_context())
        self._update_demo_events()
        self._draw_all()

    def _update_demo_events(self) -> None:
        """Connect or disconnect image-plot events depending on current demo."""
        for cid in self._demo_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._demo_cids = []
        current = get_demo_by_id(self._current_demo_id)
        if current is not None and current.needs_image_events():
            cid1 = self.canvas.mpl_connect("button_press_event", self._on_image_plot_button_press)
            cid2 = self.canvas.mpl_connect("motion_notify_event", self._on_image_plot_motion)
            cid3 = self.canvas.mpl_connect("button_release_event", self._on_image_plot_button_release)
            self._demo_cids = [cid1, cid2, cid3]

    def _on_image_plot_button_press(self, event) -> None:
        if event.inaxes != self.ax_img:
            return
        current = get_demo_by_id(self._current_demo_id)
        if current is None or not current.needs_image_events():
            return
        ctx = self._get_context()
        current.on_image_button_press(event, ctx)
        self._draw_all()

    def _on_image_plot_motion(self, event) -> None:
        if event.inaxes != self.ax_img or event.xdata is None or event.ydata is None:
            return
        current = get_demo_by_id(self._current_demo_id)
        if current is None or not current.needs_image_events():
            return
        u_d, v_d = float(event.xdata), float(event.ydata)
        dist = self.state.get_distortion()
        K = self.state.get_K()
        if distortion.distortion_params_nonzero(*dist):
            u, v = distortion.undistort_point(u_d, v_d, K, *dist)
        else:
            u, v = u_d, v_d
        # Update point(s) for demos that support dragging
        if self._current_demo_id == "backproject" and getattr(current, "is_dragging", lambda: False)():
            current.set_point_uv(u, v)
        elif self._current_demo_id == "angulometer":
            idx = getattr(current, "get_dragging_index", lambda: None)()
            if idx is not None:
                current.set_point_uv(idx, u, v)
        current.on_image_motion(event, self._get_context())
        self._draw_all()

    def _on_image_plot_button_release(self, event) -> None:
        current = get_demo_by_id(self._current_demo_id)
        if current is not None and current.needs_image_events():
            current.on_image_button_release(event, self._get_context())
        self._draw_all()

    def _set_3d_axes_limits_once(self) -> None:
        margin = 6.0
        self.ax3d.set_xlim(-margin, margin)
        self.ax3d.set_ylim(-margin, margin)
        self.ax3d.set_zlim(-0.5, margin)
        self.ax3d.set_box_aspect((1, 1, 2))

    def _on_params_changed(self) -> None:
        self.params_widget.apply_to_state()
        self._update_matrix_displays()
        self._draw_all()

    def _on_distortion_changed(self) -> None:
        self.distortion_widget.apply_to_state()
        self._draw_all()

    def _on_pose_changed(self) -> None:
        self.rotation_widget.apply_to_state()
        self.C_widget.apply_to_state()
        self._update_matrix_displays()
        self._draw_all()

    def _on_P_changed(self, P: np.ndarray) -> None:
        try:
            P = np.asarray(P).reshape(3, 4)
            if not np.allclose(P, self.state.get_P(), rtol=1e-9, atol=1e-12):
                self.state.set_from_P(P)
                self.params_widget.sync_from_state()
                self.rotation_widget.sync_from_state()
                self.C_widget.sync_from_state()
                self._update_matrix_displays()
                self._draw_all()
        except Exception:
            pass

    def _on_K_changed(self, K: np.ndarray) -> None:
        try:
            K = np.asarray(K).reshape(3, 3)
            if not np.allclose(K, self.state.get_K(), rtol=1e-9, atol=1e-12):
                self.state.set_from_K(K)
                self.params_widget.sync_from_state()
                self.rotation_widget.sync_from_state()
                self.C_widget.sync_from_state()
                self._update_matrix_displays()
                self._draw_all()
        except Exception:
            pass

    def _update_matrix_displays(self) -> None:
        P = self.state.get_P()
        K = self.state.get_K()
        R_world_to_cam, t_world_to_cam = self.state.get_R_and_t()
        self.edit_P.set_matrix(P)
        self.edit_K.set_matrix(K)
        self.display_R.set_matrix(R_world_to_cam)
        self.C_widget.sync_from_state()
        self.display_t.set_matrix(t_world_to_cam)

    def _draw_all(self) -> None:
        self.image_width_px = max(1, int(self.state.sensor_width_mm / self.state.pixel_size_x_mm))
        self.image_height_px = max(1, int(self.state.sensor_height_mm / self.state.pixel_size_y_mm))
        xlim = self.ax3d.get_xlim()
        ylim = self.ax3d.get_ylim()
        zlim = self.ax3d.get_zlim()
        self.ax3d.cla()
        self.ax_img.cla()

        camera_center_world = self.state.get_camera_center_world()
        R_cam = self.state.get_R_cw()
        P = self.state.get_P()
        base_world, apex = geometry.get_camera_pyramid(
            camera_center_world,
            R_cam,
            sensor_width_mm=self.state.sensor_width_mm,
            sensor_height_mm=self.state.sensor_height_mm,
            focal_length_mm=self.state.focal_length_mm,
        )
        verts_pyramid = [
            [apex, base_world[0], base_world[1]],
            [apex, base_world[1], base_world[2]],
            [apex, base_world[2], base_world[3]],
            [apex, base_world[3], base_world[0]],
            [base_world[0], base_world[1], base_world[2], base_world[3]],
        ]
        self.ax3d.add_collection3d(
            Poly3DCollection(verts_pyramid, facecolors="cyan", edgecolors="blue", alpha=0.4)
        )
        self.ax3d.scatter(
            self.square_pts[:, 0], self.square_pts[:, 1], self.square_pts[:, 2], c="green", s=20
        )
        self.ax3d.scatter(
            self.triangle_pts[:, 0], self.triangle_pts[:, 1], self.triangle_pts[:, 2], c="red", s=20
        )
        self.ax3d.scatter(
            self.rectangle_pts[:, 0],
            self.rectangle_pts[:, 1],
            self.rectangle_pts[:, 2],
            c="blue",
            s=20,
        )
        self.ax3d.scatter([0], [0], [0], c="purple", s=80, zorder=5, edgecolors="white", linewidths=1.5)
        sq_edges = np.vstack([self.square_pts, self.square_pts[0:1]])
        tri_edges = np.vstack([self.triangle_pts, self.triangle_pts[0:1]])
        rect_edges = np.vstack([self.rectangle_pts, self.rectangle_pts[0:1]])
        self.ax3d.plot(sq_edges[:, 0], sq_edges[:, 1], sq_edges[:, 2], "g-", lw=2)
        self.ax3d.plot(tri_edges[:, 0], tri_edges[:, 1], tri_edges[:, 2], "r-", lw=2)
        self.ax3d.plot(rect_edges[:, 0], rect_edges[:, 1], rect_edges[:, 2], "b-", lw=2)
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.set_title("3D scene and camera")
        self.ax3d.set_xlim(xlim)
        self.ax3d.set_ylim(ylim)
        self.ax3d.set_zlim(zlim)
        rx, ry, rz = xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]
        self.ax3d.set_box_aspect((rx, ry, rz))

        context = self._get_context()
        current_demo = get_demo_by_id(self._current_demo_id)
        if current_demo is not None:
            current_demo.on_draw_3d(self.ax3d, context)

        K = self.state.get_K()
        dist = self.state.get_distortion()
        use_affine = self._current_demo_id == "affine"
        rendering.draw_projected_scene(
            self.ax_img,
            P,
            self.square_pts,
            self.triangle_pts,
            self.rectangle_pts,
            self.image_width_px,
            self.image_height_px,
            K=K,
            dist=dist,
            use_affine=use_affine,
        )
        if not use_affine:
            rendering.draw_vanishing_points(
                self.ax_img, K, R_cam, self.image_width_px, self.image_height_px, dist=dist
            )
        rendering.draw_world_origin_on_image(
            self.ax_img, P, self.image_width_px, self.image_height_px, K=K, dist=dist, affine=use_affine
        )
        if current_demo is not None:
            current_demo.on_draw_image(self.ax_img, context)

        self.ax_img.set_title("Image")
        self.ax_img.set_xlabel("u (pixels)")
        self.ax_img.set_ylabel("v (pixels)")
        self._update_matrix_displays()
        self.canvas.draw_idle()


def run_app() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
