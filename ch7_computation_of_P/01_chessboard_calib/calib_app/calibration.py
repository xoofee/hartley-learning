"""
Calibration: chessboard params input, run OpenCV calibrateCamera, output K/dist/R/t to log.

Single responsibility: chessboard detection and camera calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QGroupBox,
    QFormLayout,
)

from .state import AppState, CalibrationResult, ChessboardParams
from . import logging_ui


def find_chessboard_corners(
    image: np.ndarray,
    cols: int,
    rows: int,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Find inner chessboard corners. Returns (corners, found).
    corners: (N, 1, 2) in image coords if found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    return corners, found


def run_calibration(
    image_paths: List[Path],
    chessboard: ChessboardParams,
) -> Optional[CalibrationResult]:
    """
    Run OpenCV calibrateCamera on the given images. Returns CalibrationResult or None on failure.
    """
    if not image_paths:
        return None
    cols, rows = chessboard.cols, chessboard.rows
    square_size = chessboard.square_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []
    used_paths = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        corners, found = find_chessboard_corners(img, cols, rows)
        if not found:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        used_paths.append(path)

    if len(obj_points) < 2:
        return None

    h, w = cv2.imread(str(used_paths[0])).shape[:2]
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        (w, h),
        None,
        None,
    )
    return CalibrationResult(
        K=K,
        dist=dist.ravel(),
        rvecs=rvecs,
        tvecs=tvecs,
        reproj_err=ret,
        image_paths=used_paths,
    )


def log_calibration_result(result: CalibrationResult) -> None:
    """Write K, dist, per-image R/t and reproj error to the log sink."""
    log = logging_ui.log
    log("--- Calibration result ---")
    log(f"Reprojection error: {result.reproj_err:.6f}")
    log("K (3x3):")
    for row in result.K:
        log("  " + " ".join(f"{x:.4f}" for x in row))
    d = result.dist.ravel()
    log("Distortion (k1, k2, p1, p2, k3, ...):")
    log("  " + " ".join(f"{x:.6f}" for x in d[: min(8, len(d))]))
    for i, (rvec, tvec) in enumerate(zip(result.rvecs, result.tvecs)):
        R, _ = cv2.Rodrigues(rvec)
        log(f"Image {i + 1} ({result.image_paths[i].name}):")
        log("  R (3x3):")
        for row in R:
            log("    " + " ".join(f"{x:.4f}" for x in row))
        log("  t: " + " ".join(f"{x:.4f}" for x in tvec.ravel()))
    log("--- End calibration ---")


class ChessboardParamsWidget(QWidget):
    """Inputs for chessboard cols, rows, square size."""

    def __init__(self, chessboard: ChessboardParams, parent=None):
        super().__init__(parent)
        self._params = chessboard
        layout = QFormLayout(self)
        self._spin_cols = QSpinBox()
        self._spin_cols.setRange(2, 24)
        self._spin_cols.setValue(chessboard.cols)
        layout.addRow("Inner corners (cols):", self._spin_cols)
        self._spin_rows = QSpinBox()
        self._spin_rows.setRange(2, 24)
        self._spin_rows.setValue(chessboard.rows)
        layout.addRow("Inner corners (rows):", self._spin_rows)
        self._spin_square = QDoubleSpinBox()
        self._spin_square.setRange(0.01, 1000.0)
        self._spin_square.setDecimals(4)
        self._spin_square.setValue(chessboard.square_size)
        layout.addRow("Square size (world units):", self._spin_square)

    def apply_to_params(self) -> None:
        self._params.cols = self._spin_cols.value()
        self._params.rows = self._spin_rows.value()
        self._params.square_size = self._spin_square.value()

    def sync_from_params(self) -> None:
        self._spin_cols.setValue(self._params.cols)
        self._spin_rows.setValue(self._params.rows)
        self._spin_square.setValue(self._params.square_size)


class CalibrationWidget(QWidget):
    """Chessboard params + Calibrate button; runs calibration and logs result."""

    calibration_done = pyqtSignal()

    def __init__(self, state: AppState, get_gallery_paths: callable, parent=None):
        super().__init__(parent)
        self._state = state
        self._get_gallery_paths = get_gallery_paths
        layout = QVBoxLayout(self)
        group = QGroupBox("Chessboard")
        group_layout = QVBoxLayout()
        self._params_widget = ChessboardParamsWidget(state.chessboard)
        group_layout.addWidget(self._params_widget)
        group.setLayout(group_layout)
        layout.addWidget(group)
        self._calibrate_btn = QPushButton("Calibrate from gallery")
        self._calibrate_btn.clicked.connect(self._run_calibration)
        layout.addWidget(self._calibrate_btn)
        layout.addStretch()

    def _run_calibration(self) -> None:
        self._params_widget.apply_to_params()
        paths = self._get_gallery_paths()
        if not paths:
            logging_ui.log("No images in gallery. Add images first.")
            return
        logging_ui.log(f"Calibrating with {len(paths)} images...")
        result = run_calibration(paths, self._state.chessboard)
        if result is None:
            logging_ui.log("Calibration failed (need at least 2 images with detected chessboards).")
            return
        self._state.calibration = result
        log_calibration_result(result)
        self.calibration_done.emit()
