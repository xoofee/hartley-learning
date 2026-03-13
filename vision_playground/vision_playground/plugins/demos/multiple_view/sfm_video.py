"""
SFM from video: two-view reconstruction using the current active video in the center.
Option to use calibration K (calibrated SFM) or not (uncalibrated SFM).
"""
from __future__ import annotations

from typing import Optional

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

from ...registry import Demo
from .... import logging_ui

from .feature_matching import get_supported_feature_types, match_features
from .two_view_geometry import (
    estimate_E,
    estimate_F,
    fundamental_from_E,
    point_colors,
    projection_matrices,
    recover_pose,
    triangulate,
)
from .two_view_reconstruction import TwoView3DWidget, MatchesViewWidget

_SETTINGS_ORG = "hartley-learning"
_SETTINGS_APP = "chessboard_calib"
_SETTINGS_GROUP = "sfm_video"
_DEFAULT_FEATURE_TYPE = "orb"


def _center_crop(img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Center-crop image to (target_height, target_width). Image is (H, W) or (H, W, C)."""
    h, w = img.shape[:2]
    if target_width <= 0 or target_height <= 0 or w < target_width or h < target_height:
        return img
    x0 = (w - target_width) // 2
    y0 = (h - target_height) // 2
    return np.ascontiguousarray(img[y0 : y0 + target_height, x0 : x0 + target_width])


def _K_for_center_crop(
    K: np.ndarray, calib_w: int, calib_h: int, crop_w: int, crop_h: int
) -> np.ndarray:
    """
    Return K' such that projecting with K' on the cropped image (crop_w × crop_h)
    is equivalent to center-cropping the calibration image (calib_w × calib_h) to (crop_w, crop_h) and using K.
    Principal point shifts by the crop offset; fx, fy unchanged.
    """
    K = np.asarray(K, dtype=np.float64)
    ox = (calib_w - crop_w) / 2.0
    oy = (calib_h - crop_h) / 2.0
    K_out = K.copy()
    K_out[0, 2] = K[0, 2] - ox
    K_out[1, 2] = K[1, 2] - oy
    return K_out


def _settings_bool(settings: QSettings, key: str, default: bool) -> bool:
    v = settings.value(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).lower() not in ("0", "false", "no", "")


class SfmVideoDemo(Demo):
    """SFM from current active video: two frames (start + mid) → calibrated or uncalibrated two-view reconstruction."""

    def id(self) -> str:
        return "sfm_video"

    def label(self) -> str:
        return "SFM (video)"

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
        self._feature_type_combo.setToolTip("Feature detector for matching.")
        self._feature_type_combo.currentIndexChanged.connect(self._save_demo_settings)
        form.addRow("Feature type:", self._feature_type_combo)
        layout.addLayout(form)
        self._use_k_cb = QCheckBox("Use calibration K")
        self._use_k_cb.setChecked(True)
        self._use_k_cb.setToolTip("If unchecked, use fundamental matrix (uncalibrated SFM).")
        self._use_k_cb.stateChanged.connect(self._save_demo_settings)
        layout.addWidget(self._use_k_cb)
        btn = QPushButton("Reconstruct")
        btn.setToolTip("Use current active video in center. Extracts frame at start and at 50% → two-view reconstruction.")
        btn.clicked.connect(lambda: self._run_reconstruction(context))
        layout.addWidget(btn)
        layout.addWidget(QLabel("Visualization (after Reconstruct):"))
        self._show_match_lines_cb = QCheckBox("Show match connection lines")
        self._show_match_lines_cb.setChecked(False)
        self._show_match_lines_cb.stateChanged.connect(self._on_show_match_lines_changed)
        layout.addWidget(self._show_match_lines_cb)
        self._show_inlier_outlier_cb = QCheckBox("Show inlier / outlier")
        self._show_inlier_outlier_cb.setChecked(True)
        self._show_inlier_outlier_cb.stateChanged.connect(self._on_show_inlier_outlier_changed)
        layout.addWidget(self._show_inlier_outlier_cb)
        self._show_epipolar_cb = QCheckBox("Show epipolar lines")
        self._show_epipolar_cb.setChecked(True)
        self._show_epipolar_cb.stateChanged.connect(self._on_show_epipolar_changed)
        layout.addWidget(self._show_epipolar_cb)
        self._show_reproj_error_cb = QCheckBox("Show reprojection error")
        self._show_reproj_error_cb.setChecked(True)
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
        get_video_doc = context.get("get_current_video_document")
        add_center_tab = context.get("add_center_tab")
        get_K = context.get("get_K")
        if not callable(get_video_doc) or not callable(add_center_tab):
            QMessageBox.warning(None, "SFM (video)", "Missing context.")
            return
        video_doc = get_video_doc()
        if video_doc is None:
            QMessageBox.warning(
                None,
                "SFM (video)",
                "No video open in the center. Open a video from the Video gallery (click a thumbnail).",
            )
            return
        img1 = video_doc.get_frame_at_ms(0)
        duration = video_doc.duration_ms()
        mid_ms = duration // 2 if duration > 0 else 0
        img2 = video_doc.get_frame_at_ms(mid_ms) if mid_ms > 0 else video_doc.frame_bgr()
        if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
            QMessageBox.warning(None, "SFM (video)", "Could not read frames from the video.")
            return
        use_k = self._use_k_cb.isChecked()
        K = get_K() if callable(get_K) else None
        get_calib_size = context.get("get_calibration_image_size")
        calib_size = get_calib_size() if callable(get_calib_size) else None
        if use_k and (K is None or K.size == 0):
            QMessageBox.warning(
                None,
                "SFM (video)",
                "Use calibration K is checked but no calibration (K) available. Calibrate or uncheck for uncalibrated SFM.",
            )
            return
        if use_k and K is not None and calib_size is not None:
            h, w = img1.shape[:2]
            cw, ch = calib_size
            if (w, h) != (cw, ch):
                if w >= cw and h >= ch:
                    img1 = _center_crop(img1, cw, ch)
                    img2 = _center_crop(img2, cw, ch)
                    logging_ui.log_info(
                        f"SFM (video): video size {w}×{h} differs from calibration image size {cw}×{ch}; using center crop to {cw}×{ch}."
                    )
                else:
                    K = _K_for_center_crop(K, cw, ch, w, h)
                    logging_ui.log_info(
                        f"SFM (video): video size {w}×{h} is smaller than calibration image size {cw}×{ch}; transformed K for center-crop effect (cx, cy shifted)."
                    )

        feature_type = self._get_feature_type()
        pts1, pts2, _ = match_features(img1, img2, feature_type, nfeatures=5000)
        if pts1.shape[0] < 8:
            QMessageBox.warning(None, "SFM (video)", "Too few matches between the two frames.")
            return

        R, t, mask = None, None, None
        F = None
        E = None
        if use_k and K is not None:
            E, mask = estimate_E(pts1, pts2, K)
            if E is None:
                QMessageBox.warning(None, "SFM (video)", "Essential matrix estimation failed.")
                return
            R, t, pose_mask = recover_pose(E, pts1, pts2, K)
            if R is None:
                QMessageBox.warning(None, "SFM (video)", "Pose recovery failed.")
                return
            mask = (mask.astype(bool) & (pose_mask.astype(bool))).astype(np.uint8)
            F = fundamental_from_E(E, K)
        else:
            F, mask = estimate_F(pts1, pts2)
            if F is None:
                QMessageBox.warning(None, "SFM (video)", "Fundamental matrix estimation failed.")
                return
            mask = mask.astype(np.uint8)

        P1, P2 = projection_matrices(K, R, t, F)
        pts3d = triangulate(P1, P2, pts1, pts2, mask)
        if pts3d.shape[1] == 0:
            QMessageBox.warning(None, "SFM (video)", "No inlier triangulated points.")
            return
        colors = point_colors(img1, pts1, mask)

        if R is None or t is None:
            R = np.eye(3)
            t = np.zeros(3)
        t_flat = t.ravel() if hasattr(t, "ravel") else np.asarray(t).ravel()

        if not hasattr(self, "_recon3d_widget") or self._recon3d_widget is None:
            self._recon3d_widget = TwoView3DWidget()
        self._recon3d_widget.set_scene(pts3d, colors, R, t_flat, K)
        add_center_tab(self._recon3d_widget, "3D Reconstruction (video)")

        if not hasattr(self, "_matches_widget") or self._matches_widget is None:
            self._matches_widget = MatchesViewWidget()
        self._matches_widget.set_data(img1, img2, pts1, pts2, mask, F, P1, P2, pts3d)
        self._matches_widget.set_show_match_lines(self._show_match_lines_cb.isChecked())
        self._matches_widget.set_show_inlier_outlier(self._show_inlier_outlier_cb.isChecked())
        self._matches_widget.set_show_epipolar(self._show_epipolar_cb.isChecked())
        self._matches_widget.set_show_reproj_error(self._show_reproj_error_cb.isChecked())
        add_center_tab(self._matches_widget, "Matches & Epipolar (video)")
        self._save_demo_settings()
