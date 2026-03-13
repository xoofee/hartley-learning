from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QCheckBox,
    QLabel,
    QComboBox,
    QFormLayout,
)

from ....plot3d import camera_pyramid_from_K, draw_axes
from ....widgets import ImageViewWidget
from ...registry import Demo

from .feature_matching import get_supported_feature_types, match_features
from .two_view_geometry import (
    camera_poses_from_Rt,
    compute_epilines,
    default_K_for_pyramid,
    estimate_E,
    estimate_F,
    fundamental_from_E,
    point_colors,
    projection_matrices,
    recover_pose,
    reprojection_error,
    triangulate,
)

# Same as app.py for persistence (demo reads/writes its own keys under a group)
_SETTINGS_ORG = "hartley-learning"
_SETTINGS_APP = "chessboard_calib"
_SETTINGS_GROUP = "two_view_reconstruction"
_DEFAULT_FEATURE_TYPE = "orb"


def _settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    v = settings.value(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).lower() not in ("0", "false", "no", "")


# ---------------------------------------------------------------------------
# 3D and matches widgets (use geometry module for math)
# ---------------------------------------------------------------------------


class TwoView3DWidget(QWidget):
    """Matplotlib 3D widget: scatter point cloud + two camera pyramids."""

    def __init__(self, parent=None):
        super().__init__(parent)
        import matplotlib
        matplotlib.use("Qt5Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        self._fig = plt.figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)
        self._empty = True

    def set_scene(
        self,
        pts3d: np.ndarray,
        colors: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: Optional[np.ndarray] = None,
    ) -> None:
        """pts3d (3, N), colors (N, 3) [0,1], R (3,3) t (3,) for second camera; first at origin. K for pyramid aspect."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        self._ax.cla()
        K_use = K if K is not None else default_K_for_pyramid()
        t_arr = np.asarray(t).reshape(3, 1) if np.asarray(t).size == 3 else np.asarray(t)
        (R1, t1), (R2, t2) = camera_poses_from_Rt(R, t_arr)

        if pts3d.size > 0:
            self._ax.scatter(pts3d[0], pts3d[1], pts3d[2], c=colors, s=1, rasterized=True)
        scale = 0.15
        for R_cw, t_cw, facecolor, edgecolor in [
            (R1, t1, "cyan", "blue"),
            (R2, t2, "orange", "darkorange"),
        ]:
            base_w, apex_w = camera_pyramid_from_K(R_cw, t_cw, K_use, height=0.04 * scale * 10, scale=scale)
            verts = [
                [apex_w, base_w[0], base_w[1]],
                [apex_w, base_w[1], base_w[2]],
                [apex_w, base_w[2], base_w[3]],
                [apex_w, base_w[3], base_w[0]],
                [base_w[0], base_w[1], base_w[2], base_w[3]],
            ]
            self._ax.add_collection3d(
                Poly3DCollection(verts, facecolors=facecolor, edgecolors=edgecolor, alpha=0.5)
            )
            draw_axes(self._ax, R_cw, t_cw, length=scale * 0.5)
        self._ax.set_xlabel("X")
        self._ax.set_ylabel("Y")
        self._ax.set_zlabel("Z")
        # self._ax.set_title("Two-view reconstruction")
        self._ax.set_box_aspect((1, 1, 1))
        if pts3d.size > 0:
            c = pts3d.mean(axis=1)
            r = max(1e-6, float(np.ptp(pts3d) / 2))
            self._ax.set_xlim(c[0] - r, c[0] + r)
            self._ax.set_ylim(c[1] - r, c[1] + r)
            self._ax.set_zlim(c[2] - r, c[2] + r)
        self._canvas.draw_idle()
        self._empty = False


# ---------------------------------------------------------------------------
# Matches & epipolar visualization (render only; display uses ImageViewWidget)
# ---------------------------------------------------------------------------

_MAX_EPILINES = 50
_MAX_MATCH_LINES = 200


def render_matches_canvas(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    mask: np.ndarray,
    *,
    F: Optional[np.ndarray] = None,
    reproj_error: Optional[np.ndarray] = None,
    show_inlier_outlier: bool = True,
    show_epipolar: bool = True,
    show_match_lines: bool = False,
    show_reproj_error: bool = True,
) -> np.ndarray:
    """
    Render side-by-side matches view (img1 | img2) with optional overlays.
    Returns BGR image (H, W, 3) for use with ImageViewWidget.set_image().
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    w_combined = w1 + w2
    canvas = np.zeros((h, w_combined, 3), dtype=np.uint8)
    canvas[:] = 128
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2

    inliers1 = pts1[mask == 1]
    inliers2 = pts2[mask == 1]
    outliers1 = pts1[mask == 0]
    outliers2 = pts2[mask == 0]

    if show_inlier_outlier:
        for x, y in inliers1.astype(int):
            cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)
        for x, y in outliers1.astype(int):
            cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
        for x, y in inliers2.astype(int):
            cv2.circle(canvas, (w1 + x, y), 3, (0, 255, 0), -1)
        for x, y in outliers2.astype(int):
            cv2.circle(canvas, (w1 + x, y), 3, (0, 0, 255), -1)

    if show_reproj_error and reproj_error is not None and reproj_error.size > 0:
        err = reproj_error
        err_min, err_max = float(err.min()), float(err.max())
        err_n = (err - err_min) / (err_max - err_min) if err_max > err_min else np.zeros_like(err)
        for i, (x, y) in enumerate(inliers1.astype(int)):
            t = err_n[i]
            r, g, b = int(255 * (1 - t)), 128, int(255 * t)
            cv2.circle(canvas, (x, y), 5, (b, g, r), 2)
        for i, (x, y) in enumerate(inliers2.astype(int)):
            t = err_n[i]
            r, g, b = int(255 * (1 - t)), 128, int(255 * t)
            cv2.circle(canvas, (w1 + x, y), 5, (b, g, r), 2)

    if show_epipolar and F is not None and inliers1.shape[0] > 0:
        n_show = min(_MAX_EPILINES, inliers1.shape[0])
        idx = np.linspace(0, inliers1.shape[0] - 1, n_show, dtype=int)
        pts_sample = inliers1[idx]
        lines2 = compute_epilines(pts_sample, F, 1)
        for k, i in enumerate(idx):
            line = lines2[k]
            pt2 = inliers2[i]
            a, b, c = line[0], line[1], line[2]
            if abs(b) < 1e-6:
                x0 = x1 = int(-c / a) if abs(a) > 1e-6 else 0
                y0, y1 = 0, h2 - 1
            else:
                x0, x1 = 0, w2 - 1
                y0 = int(-(c + a * x0) / b)
                y1 = int(-(c + a * x1) / b)
            x0 = max(0, min(w2 - 1, x0))
            x1 = max(0, min(w2 - 1, x1))
            y0 = max(0, min(h2 - 1, y0))
            y1 = max(0, min(h2 - 1, y1))
            cv2.line(canvas, (w1 + x0, y0), (w1 + x1, y1), (0, 255, 255), 1)
            cv2.circle(canvas, (w1 + int(pt2[0]), int(pt2[1])), 4, (0, 0, 255), -1)

    if show_match_lines:
        in1 = pts1[mask == 1]
        in2 = pts2[mask == 1]
        n_lines = min(_MAX_MATCH_LINES, in1.shape[0])
        idx = np.linspace(0, in1.shape[0] - 1, n_lines, dtype=int)
        for i in idx:
            x1, y1 = int(in1[i, 0]), int(in1[i, 1])
            x2, y2 = int(in2[i, 0]), int(in2[i, 1])
            cv2.line(canvas, (x1, y1), (w1 + x2, y2), (255, 128, 0), 1)

    return np.ascontiguousarray(canvas)


class MatchesViewWidget(QWidget):
    """Side-by-side matches view with zoom/pan via shared ImageViewWidget. Holds data and options; rendering is delegated to render_matches_canvas()."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._image_view = ImageViewWidget(self)
        self._image_view.set_image(None, placeholder="Run Reconstruct to see matches")
        layout.addWidget(self._image_view)

        self._img1: Optional[np.ndarray] = None
        self._img2: Optional[np.ndarray] = None
        self._pts1: Optional[np.ndarray] = None
        self._pts2: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._F: Optional[np.ndarray] = None
        self._reproj_error: Optional[np.ndarray] = None
        self._show_inlier_outlier = True
        self._show_epipolar = True
        self._show_match_lines = False
        self._show_reproj_error = True

    def set_data(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        mask: np.ndarray,
        F: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
        pts3d: np.ndarray,
    ) -> None:
        self._img1 = img1.copy() if img1 is not None else None
        self._img2 = img2.copy() if img2 is not None else None
        self._pts1 = np.asarray(pts1)
        self._pts2 = np.asarray(pts2)
        self._mask = np.asarray(mask)
        self._F = np.asarray(F) if F is not None else None
        if pts3d is not None and pts3d.size > 0 and P1 is not None and P2 is not None:
            in1 = pts1[mask == 1]
            in2 = pts2[mask == 1]
            self._reproj_error = reprojection_error(P1, P2, in1, in2, pts3d)
        else:
            self._reproj_error = None
        self._redraw(preserve_view=False)

    def set_show_inlier_outlier(self, on: bool) -> None:
        self._show_inlier_outlier = on
        self._redraw(preserve_view=True)

    def set_show_epipolar(self, on: bool) -> None:
        self._show_epipolar = on
        self._redraw(preserve_view=True)

    def set_show_match_lines(self, on: bool) -> None:
        self._show_match_lines = on
        self._redraw(preserve_view=True)

    def set_show_reproj_error(self, on: bool) -> None:
        self._show_reproj_error = on
        self._redraw(preserve_view=True)

    def _redraw(self, preserve_view: bool = True) -> None:
        if self._img1 is None or self._img2 is None or self._pts1 is None or self._pts2 is None:
            return
        canvas_bgr = render_matches_canvas(
            self._img1,
            self._img2,
            self._pts1,
            self._pts2,
            self._mask,
            F=self._F,
            reproj_error=self._reproj_error,
            show_inlier_outlier=self._show_inlier_outlier,
            show_epipolar=self._show_epipolar,
            show_match_lines=self._show_match_lines,
            show_reproj_error=self._show_reproj_error,
        )
        self._image_view.set_image(canvas_bgr, placeholder="No matches", preserve_view=preserve_view)


class TwoViewReconstructionDemo(Demo):
    """Two-view SfM: match → F/E → pose → triangulate → 3D tab with point cloud and camera poses."""

    def id(self) -> str:
        return "two_view_reconstruction"

    def label(self) -> str:
        return "Two-view reconstruction"

    def hide_calibration_pyramids(self) -> bool:
        return True

    def on_activated(self, context: dict) -> None:
        self._context = context

    def on_deactivated(self) -> None:
        self._context = None

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self._feature_type_combo = QComboBox()
        for ft in get_supported_feature_types():
            self._feature_type_combo.addItem(ft.upper(), ft)
        self._feature_type_combo.setToolTip("Feature detector for matching (ORB fast, SIFT more accurate).")
        self._feature_type_combo.currentIndexChanged.connect(self._on_feature_type_changed)
        form.addRow("Feature type:", self._feature_type_combo)
        layout.addLayout(form)
        self._use_k_cb = QCheckBox("Use calibration K (essential matrix)")
        self._use_k_cb.setChecked(True)
        self._use_k_cb.setToolTip("If unchecked, use fundamental matrix (uncalibrated).")
        self._use_k_cb.stateChanged.connect(self._on_use_k_changed)
        layout.addWidget(self._use_k_cb)
        btn = QPushButton("Reconstruct")
        btn.setToolTip("Requires exactly 2 images open in the center. Match → F/E → pose → triangulate → 3D tab.")
        btn.clicked.connect(lambda: self._run_reconstruction(context))
        layout.addWidget(btn)
        layout.addWidget(QLabel("Visualization (after Reconstruct):"))
        self._show_match_lines_cb = QCheckBox("Show match connection lines")
        self._show_match_lines_cb.setChecked(False)
        self._show_match_lines_cb.setToolTip("Draw lines between matched ORB features across the two images.")
        self._show_match_lines_cb.stateChanged.connect(self._on_show_match_lines_changed)
        layout.addWidget(self._show_match_lines_cb)
        self._show_inlier_outlier_cb = QCheckBox("Show inlier / outlier")
        self._show_inlier_outlier_cb.setChecked(True)
        self._show_inlier_outlier_cb.setToolTip("Green = inliers, Red = outliers (RANSAC).")
        self._show_inlier_outlier_cb.stateChanged.connect(self._on_show_inlier_outlier_changed)
        layout.addWidget(self._show_inlier_outlier_cb)
        self._show_epipolar_cb = QCheckBox("Show epipolar lines")
        self._show_epipolar_cb.setChecked(True)
        self._show_epipolar_cb.stateChanged.connect(self._on_show_epipolar_changed)
        layout.addWidget(self._show_epipolar_cb)
        self._show_reproj_error_cb = QCheckBox("Show reprojection error")
        self._show_reproj_error_cb.setChecked(True)
        self._show_reproj_error_cb.setToolTip("Color inlier points by mean reprojection error (blue=low, red=high).")
        self._show_reproj_error_cb.stateChanged.connect(self._on_show_reproj_error_changed)
        layout.addWidget(self._show_reproj_error_cb)
        layout.addStretch()
        self._restore_demo_settings()
        return widget

    def _get_feature_type(self) -> str:
        data = self._feature_type_combo.currentData()
        return str(data) if data else _DEFAULT_FEATURE_TYPE

    def _restore_demo_settings(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.beginGroup(_SETTINGS_GROUP)
        saved_ft = settings.value("feature_type", _DEFAULT_FEATURE_TYPE)
        if saved_ft:
            idx = self._feature_type_combo.findData(saved_ft)
            if idx >= 0:
                self._feature_type_combo.setCurrentIndex(idx)
        self._use_k_cb.setChecked(_settings_bool(settings, "use_k", True))
        self._show_match_lines_cb.setChecked(_settings_bool(settings, "show_match_lines", False))
        self._show_inlier_outlier_cb.setChecked(_settings_bool(settings, "show_inlier_outlier", True))
        self._show_epipolar_cb.setChecked(_settings_bool(settings, "show_epipolar", True))
        self._show_reproj_error_cb.setChecked(_settings_bool(settings, "show_reproj_error", True))
        settings.endGroup()

    def _save_demo_settings(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.beginGroup(_SETTINGS_GROUP)
        settings.setValue("feature_type", self._get_feature_type())
        settings.setValue("use_k", self._use_k_cb.isChecked())
        settings.setValue("show_match_lines", self._show_match_lines_cb.isChecked())
        settings.setValue("show_inlier_outlier", self._show_inlier_outlier_cb.isChecked())
        settings.setValue("show_epipolar", self._show_epipolar_cb.isChecked())
        settings.setValue("show_reproj_error", self._show_reproj_error_cb.isChecked())
        settings.endGroup()

    def _on_feature_type_changed(self, _index: int) -> None:
        self._save_demo_settings()

    def _on_use_k_changed(self, _state: int) -> None:
        self._save_demo_settings()

    def _on_show_match_lines_changed(self, _state: int) -> None:
        self._save_demo_settings()
        if hasattr(self, "_matches_widget") and self._matches_widget is not None:
            self._matches_widget.set_show_match_lines(self._show_match_lines_cb.isChecked())

    def _on_show_inlier_outlier_changed(self, _state: int) -> None:
        self._save_demo_settings()
        if hasattr(self, "_matches_widget") and self._matches_widget is not None:
            self._matches_widget.set_show_inlier_outlier(self._show_inlier_outlier_cb.isChecked())

    def _on_show_epipolar_changed(self, _state: int) -> None:
        self._save_demo_settings()
        if hasattr(self, "_matches_widget") and self._matches_widget is not None:
            self._matches_widget.set_show_epipolar(self._show_epipolar_cb.isChecked())

    def _on_show_reproj_error_changed(self, _state: int) -> None:
        self._save_demo_settings()
        if hasattr(self, "_matches_widget") and self._matches_widget is not None:
            self._matches_widget.set_show_reproj_error(self._show_reproj_error_cb.isChecked())

    def _run_reconstruction(self, context: dict) -> None:
        get_sorted = context.get("get_open_documents_sorted")
        add_center_tab = context.get("add_center_tab")
        get_K = context.get("get_K")
        if not callable(get_sorted) or not callable(add_center_tab):
            QMessageBox.warning(None, "Two-view reconstruction", "Missing context (get_open_documents_sorted / add_center_tab).")
            return
        images = get_sorted()
        if len(images) < 2:
            QMessageBox.warning(None, "Two-view reconstruction", "Open at least 2 images in the center (from gallery).")
            return
        if len(images) > 2:
            QMessageBox.information(None, "Two-view reconstruction", "More than 2 images open; using first two (by name).")
        (path1, img1), (path2, img2) = images[0], images[1]
        use_k = self._use_k_cb.isChecked()
        K = get_K() if callable(get_K) else None
        if use_k and (K is None or K.size == 0):
            QMessageBox.warning(None, "Two-view reconstruction", "Use calibration K is checked but no calibration result (K) available. Calibrate or uncheck.")
            return

        feature_type = self._get_feature_type()
        pts1, pts2, _ = match_features(img1, img2, feature_type, nfeatures=5000)
        if pts1.shape[0] < 8:
            QMessageBox.warning(None, "Two-view reconstruction", "Too few matches between the two images.")
            return

        R, t, mask = None, None, None
        F = None
        E = None
        if use_k and K is not None:
            E, mask = estimate_E(pts1, pts2, K)
            if E is None:
                QMessageBox.warning(None, "Two-view reconstruction", "Essential matrix estimation failed.")
                return
            R, t, pose_mask = recover_pose(E, pts1, pts2, K)
            if R is None:
                QMessageBox.warning(None, "Two-view reconstruction", "Pose recovery failed.")
                return
            mask = (mask.astype(bool) & (pose_mask.astype(bool))).astype(np.uint8)
            F = fundamental_from_E(E, K)
        else:
            F, mask = estimate_F(pts1, pts2)
            if F is None:
                QMessageBox.warning(None, "Two-view reconstruction", "Fundamental matrix estimation failed.")
                return
            mask = mask.astype(np.uint8)

        P1, P2 = projection_matrices(K, R, t, F)
        pts3d = triangulate(P1, P2, pts1, pts2, mask)
        if pts3d.shape[1] == 0:
            QMessageBox.warning(None, "Two-view reconstruction", "No inlier triangulated points.")
            return
        colors = point_colors(img1, pts1, mask)

        if R is None or t is None:
            R = np.eye(3)
            t = np.zeros(3)
        t_flat = t.ravel() if hasattr(t, "ravel") else np.asarray(t).ravel()

        tab_title_3d = "3D Reconstruction"
        if not hasattr(self, "_recon3d_widget") or self._recon3d_widget is None:
            self._recon3d_widget = TwoView3DWidget()
        self._recon3d_widget.set_scene(pts3d, colors, R, t_flat, K)
        add_center_tab(self._recon3d_widget, tab_title_3d)

        if not hasattr(self, "_matches_widget") or self._matches_widget is None:
            self._matches_widget = MatchesViewWidget()
        self._matches_widget.set_data(img1, img2, pts1, pts2, mask, F, P1, P2, pts3d)
        self._matches_widget.set_show_match_lines(self._show_match_lines_cb.isChecked())
        self._matches_widget.set_show_inlier_outlier(self._show_inlier_outlier_cb.isChecked())
        self._matches_widget.set_show_epipolar(self._show_epipolar_cb.isChecked())
        self._matches_widget.set_show_reproj_error(self._show_reproj_error_cb.isChecked())
        add_center_tab(self._matches_widget, "Matches & Epipolar")