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
    QCheckBox,
)
from PyQt5.QtCore import Qt

from . import scene
from . import state as state_module
from . import geometry
from . import rendering
from . import distortion
from .widgets import (
    MatrixDisplayWidget,
    MatrixEditWidget,
    CameraParamsWidget,
    DistortionParamsWidget,
    RotationParamsWidget,
    CameraCenterWidget,
)
from .plugins import get_features
from .plugins.builtin import register_builtin_features

# Register built-in features (P row planes, backproject) so they appear as checkboxes.
register_builtin_features()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.square_pts = scene.get_scene_square(1.0, 0.0)
        self.triangle_pts = scene.get_scene_triangle(0.8, 1.0, 1.5)
        self.rectangle_pts = scene.get_scene_rectangle(0.4, 0.4, 1.5, y_center=0.8, z_center=0.4)
        self.state = state_module.CameraState()
        self._backproject_cids: list[int] = []
        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Camera simulation (P = K [R|t])")
        self.setGeometry(80, 80, 1400, 700)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        self.fig = plt.figure(figsize=(8, 5))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_img = self.fig.add_subplot(122)
        self._set_3d_axes_limits_once()
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 1)

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

        # Plugin checkboxes (built-in: show P planes, backproject)
        self._feature_instances = {}
        for name, feat in get_features().items():
            w = feat.checkbox_widget(self)
            if w is not None:
                right_layout.addWidget(w)
                self._feature_instances[name] = feat
                if name == "show_P_planes":
                    w.stateChanged.connect(self._draw_all)
                elif name == "backproject":
                    w.stateChanged.connect(self._on_backproject_toggled)

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

    def _on_backproject_toggled(self) -> None:
        backproject = self._feature_instances.get("backproject")
        if backproject is None:
            return
        if backproject.is_checked():
            backproject.set_point_from_context(self._get_context())
            self._connect_image_plot_events()
        else:
            self._disconnect_image_plot_events()
        self._draw_all()

    def _connect_image_plot_events(self) -> None:
        self._disconnect_image_plot_events()
        cid1 = self.canvas.mpl_connect("button_press_event", self._on_image_plot_button_press)
        cid2 = self.canvas.mpl_connect("motion_notify_event", self._on_image_plot_motion)
        cid3 = self.canvas.mpl_connect("button_release_event", self._on_image_plot_button_release)
        self._backproject_cids = [cid1, cid2, cid3]

    def _disconnect_image_plot_events(self) -> None:
        for cid in self._backproject_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._backproject_cids = []

    def _on_image_plot_button_press(self, event) -> None:
        backproject = self._feature_instances.get("backproject")
        if backproject is None or not backproject.is_checked() or event.inaxes != self.ax_img:
            return
        backproject.set_dragging(True)

    def _on_image_plot_motion(self, event) -> None:
        backproject = self._feature_instances.get("backproject")
        if (
            backproject is None
            or not backproject.is_checked()
            or not backproject.is_dragging()
            or event.inaxes != self.ax_img
        ):
            return
        if event.xdata is not None and event.ydata is not None:
            u_d, v_d = float(event.xdata), float(event.ydata)
            dist = self.state.get_distortion()
            if distortion.distortion_params_nonzero(*dist):
                K = self.state.get_K()
                u, v = distortion.undistort_point(u_d, v_d, K, *dist)
            else:
                u, v = u_d, v_d
            backproject.set_point_uv(u, v)
            self._draw_all()

    def _on_image_plot_button_release(self, event) -> None:
        backproject = self._feature_instances.get("backproject")
        if backproject is not None and backproject.is_checked():
            backproject.set_dragging(False)

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
            camera_center_world, R_cam, scale=0.5, depth=0.3
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
        for feat in self._feature_instances.values():
            feat.on_draw_3d(self.ax3d, context)

        K = self.state.get_K()
        dist = self.state.get_distortion()
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
        )
        rendering.draw_vanishing_points(
            self.ax_img, K, R_cam, self.image_width_px, self.image_height_px, dist=dist
        )
        rendering.draw_world_origin_on_image(
            self.ax_img, P, self.image_width_px, self.image_height_px, K=K, dist=dist
        )
        for feat in self._feature_instances.values():
            feat.on_draw_image(self.ax_img, context)

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
