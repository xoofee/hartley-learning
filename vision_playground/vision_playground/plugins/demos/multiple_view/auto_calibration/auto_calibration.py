"""
auto calibration demo from n (3 or >=5) views

always assume:
    same camera intrinsics
    general camera motion
    scene not planar

case 1:
    only 3 view

    assume: square pixel, centeral principal point, no distortion

case 2:
    more then 4 view (>=5)

    only assume no distortion. general K

report error for only 4 view

if this is correct and feasible, implement it.

remember to make the code modular and flexible and have a good architecture. Do not make God object code. reuse and split if possible.

do not remove this comment.

"""
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
    QLabel,
    QComboBox,
    QFormLayout,
)

from ..... import logging_ui
from ....registry import Demo

from ..feature_matching import get_supported_feature_types, match_features
from ..multi_view_calibration import (
    UnsupportedViewCountError,
    calibrate_3views_from_F,
    calibrate_general_k_from_F,
    validate_view_count,
)
from ..multi_view_tracks import build_tracks_3views

_SETTINGS_ORG = "hartley-learning"
_SETTINGS_APP = "chessboard_calib"
_SETTINGS_GROUP = "auto_calibration"
_DEFAULT_FEATURE_TYPE = "orb"


def _compute_F_ransac(pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
    """RANSAC fundamental matrix. Returns F (3,3) or None."""
    F, _ = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
    )
    return F


def _run_3view(
    img1: np.ndarray,
    img2: np.ndarray,
    img3: np.ndarray,
    feature_type: str,
) -> Optional[tuple[np.ndarray, float]]:
    """Run 3-view calibration (K = diag(f,f,1)). Returns (K, f) or None."""
    logging_ui.log_debug(f"in _run_3view")

    pts1, pts2, pts3 = build_tracks_3views(img1, img2, img3, feature_type=feature_type)
    logging_ui.log_debug(f"Auto calibration 3-view: tracks count = {pts1.shape[0]}")
    if pts1.shape[0] < 8:
        logging_ui.log_debug("Auto calibration 3-view: failed (too few tracks, need >= 8)")
        return None
    F12 = _compute_F_ransac(pts1, pts2)
    F13 = _compute_F_ransac(pts1, pts3)
    if F12 is None:
        logging_ui.log_debug("Auto calibration 3-view: F12 (view 1-2) estimation failed")
    if F13 is None:
        logging_ui.log_debug("Auto calibration 3-view: F13 (view 1-3) estimation failed")
    if F12 is None or F13 is None:
        return None
    K, f = calibrate_3views_from_F(F12, F13)
    return (K, f)


def _run_5plus_views(
    images: list[np.ndarray],
    feature_type: str,
    image_size: tuple[int, int],
) -> Optional[np.ndarray]:
    """Run general K calibration from >=5 views. Returns K (3,3) or None."""
    n = len(images)
    F_list = []
    for j in range(1, n):
        pts0, ptsj, _ = match_features(images[0], images[j], feature_type, nfeatures=5000)
        if pts0.shape[0] < 8:
            logging_ui.log_debug(f"Auto calibration 5+: view 0-{j} matches {pts0.shape[0]} (< 8), skip")
            continue
        F = _compute_F_ransac(pts0, ptsj)
        if F is not None:
            F_list.append(F)
        else:
            logging_ui.log_debug(f"Auto calibration 5+: view 0-{j} F estimation failed")
    logging_ui.log_debug(f"Auto calibration 5+: got {len(F_list)} F matrices (need >= 2)")
    if len(F_list) < 2:
        return None
    K = calibrate_general_k_from_F(F_list, image_size, fx_fy_guess=500.0)
    if K is None:
        logging_ui.log_debug("Auto calibration 5+: general K solver returned None")
    return K


class AutoCalibrationDemo(Demo):
    """Auto calibration from 3 or >=5 views (same intrinsics, general motion, non-planar)."""

    def id(self) -> str:
        return "auto_calibration"

    def label(self) -> str:
        return "Auto calibration (3 or 5+ views)"

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
        form.addRow("Feature type:", self._feature_type_combo)
        layout.addLayout(form)
        btn = QPushButton("Run auto calibration")
        btn.setToolTip(
            "Use 3 views (K = diag(f,f,1)) or 5+ views (general K). "
            "Exactly 4 views will show an error."
        )
        btn.clicked.connect(lambda: self._run(context))
        layout.addWidget(btn)
        self._result_label = QLabel("Result: —")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)
        layout.addStretch()
        self._restore_settings(context)
        return widget

    def _get_feature_type(self) -> str:
        data = self._feature_type_combo.currentData()
        return str(data) if data else _DEFAULT_FEATURE_TYPE

    def _restore_settings(self, context: dict) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.beginGroup(_SETTINGS_GROUP)
        saved = settings.value("feature_type", _DEFAULT_FEATURE_TYPE)
        if saved:
            idx = self._feature_type_combo.findData(saved)
            if idx >= 0:
                self._feature_type_combo.setCurrentIndex(idx)
        settings.endGroup()

    def _save_settings(self) -> None:
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        settings.beginGroup(_SETTINGS_GROUP)
        settings.setValue("feature_type", self._get_feature_type())
        settings.endGroup()

    def _run(self, context: dict) -> None:
        logging_ui.log_debug("Auto calibration: _run started")
        get_sorted = context.get("get_open_documents_sorted")
        if not callable(get_sorted):
            logging_ui.log_warning("Auto calibration: missing context get_open_documents_sorted")
            QMessageBox.warning(
                None, "Auto calibration", "Missing context (get_open_documents_sorted)."
            )
            return
        images_with_paths = get_sorted()
        logging_ui.log_debug(f"Auto calibration: {len(images_with_paths)} images open")
        if len(images_with_paths) < 3:
            logging_ui.log_warning("Auto calibration: need at least 3 images")
            QMessageBox.warning(
                None,
                "Auto calibration",
                "Open at least 3 images in the center (from gallery).",
            )
            return
        try:
            validate_view_count(len(images_with_paths))
        except UnsupportedViewCountError as e:
            logging_ui.log_error(f"Auto calibration: unsupported view count — {e}")
            QMessageBox.warning(
                None,
                "Auto calibration",
                str(e),
            )
            self._result_label.setText("Result: Error — 4 views not supported.")
            return
        except ValueError as e:
            logging_ui.log_error(f"Auto calibration: validate error — {e}")
            QMessageBox.warning(None, "Auto calibration", str(e))
            self._result_label.setText(f"Result: Error — {e}")
            return

        images = [img for _, img in images_with_paths]
        n = len(images)
        feature_type = self._get_feature_type()
        self._save_settings()
        logging_ui.log_debug(f"Auto calibration: n={n} views, feature_type={feature_type}")

        h, w = images[0].shape[:2]
        image_size = (w, h)

        if n == 3:
            logging_ui.log_debug("Auto calibration: running 3-view path")
            out = _run_3view(images[0], images[1], images[2], feature_type)
            if out is None:
                logging_ui.log_warning("Auto calibration: 3-view failed (tracks or F)")
                self._result_label.setText(
                    "Result: Failed (too few tracks or F estimation failed)."
                )
                QMessageBox.warning(
                    None, "Auto calibration", "3-view calibration failed (matches or F)."
                )
                return
            K, f = out
            logging_ui.log_info(f"Auto calibration: 3-view OK, f={f:.1f} \n K =\n{K}")
            self._result_label.setText(
                f"Result: 3-view — K = diag(f,f,1), f = {f:.1f}\n"
                f"K =\n{K}"
            )
            QMessageBox.information(
                None,
                "Auto calibration",
                f"3-view calibration OK.\nK = diag(f, f, 1), f = {f:.1f}",
            )
            return

        # n >= 5
        logging_ui.log_debug(f"Auto calibration: running general K path ({n} views)")
        K = _run_5plus_views(images, feature_type, image_size)
        if K is None:
            logging_ui.log_warning("Auto calibration: general K failed (F list or solver)")
            self._result_label.setText(
                "Result: Failed (too few F matrices or general K solver failed)."
            )
            QMessageBox.warning(
                None, "Auto calibration", "General K calibration failed."
            )
            return
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        logging_ui.log_info(f"Auto calibration: general K OK ({n} views) fx={fx:.1f} fy={fy:.1f}")
        self._result_label.setText(
            f"Result: General K ({n} views)\nfx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}\nK =\n{K}"
        )
        QMessageBox.information(
            None,
            "Auto calibration",
            f"General K from {n} views.\nfx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}",
        )