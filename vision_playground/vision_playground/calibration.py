"""
Calibration: chessboard params input, run OpenCV calibrateCamera, output K/dist/R/t to log.

Single responsibility: chessboard detection and camera calibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from math import radians, tan
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
    QPlainTextEdit,
)

from .state import AppState, CalibrationResult, ChessboardParams
from . import logging_ui


def reorder_corners_if_needed(corners: np.ndarray) -> np.ndarray:
    """Reorder corners so they start from top-left (first corner has smaller x,y than last)."""
    pts = corners.reshape(-1, 2)
    first = pts[0]
    last = pts[-1]
    # If first corner is lower-right of last corner, we are flipped.
    if first[0] > last[0] and first[1] > last[1]:
        pts = pts[::-1]
    return pts.reshape(-1, 1, 2)


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
        corners = reorder_corners_if_needed(corners)
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
        image_size=(w, h),
    )


def log_calibration_result(result: CalibrationResult) -> None:
    """Write K, dist, image size, per-image R/t and reproj error to the log sink."""
    log = logging_ui.log
    log("--- Calibration result ---")
    log(f"Reprojection error: {result.reproj_err:.6f}")
    log("K (3×3):")
    for row in result.K:
        log("  " + "  ".join(f"{x:.4f}" for x in row))
    d = result.dist.ravel()
    log("Distortion (k1, k2, p1, p2, k3, ...):")
    log("  " + "  ".join(f"{x:.6f}" for x in d[: min(8, len(d))]))
    log("Image size:")
    log(f"  {result.image_size[0]}×{result.image_size[1]}" if result.image_size else "  ...")
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


def fake_K_from_image_size(w: int, h: int, horizontal_fov_deg: float = 80.0) -> np.ndarray:
    """Build intrinsic matrix K centered on image with given horizontal FOV (no distortion).
    K has principal point at (w/2, h/2); fx = (w/2)/tan(h_fov/2), fy = fx (square pixels)."""
    cx, cy = w / 2.0, h / 2.0
    half_fov_rad = radians(horizontal_fov_deg / 2.0)
    fx = (w / 2.0) / tan(half_fov_rad)
    fy = fx
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


class CalibrationWidget(QWidget):
    """Chessboard params + Calibrate button; runs calibration and logs result."""

    calibration_done = pyqtSignal()

    def __init__(
        self,
        state: AppState,
        get_gallery_paths: Callable[[], List[Path]],
        get_current_image_size: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._state = state
        self._get_gallery_paths = get_gallery_paths
        self._get_current_image_size = get_current_image_size
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
        # Fake K from current image (temporary, not persisted)
        fake_group = QGroupBox("Fake K (temporary)")
        fake_layout = QFormLayout()
        self._fake_fov_spin = QDoubleSpinBox()
        self._fake_fov_spin.setRange(1.0, 179.0)
        self._fake_fov_spin.setValue(80.0)
        self._fake_fov_spin.setSuffix(" °")
        self._fake_fov_spin.setToolTip("Horizontal field of view in degrees")
        fake_layout.addRow(QLabel("Horizontal FOV:"), self._fake_fov_spin)
        self._fake_k_btn = QPushButton("Fake K from image")
        self._fake_k_btn.setToolTip("Generate K from the current center image size and FOV. Overwrites K in memory (not saved).")
        self._fake_k_btn.clicked.connect(self._on_fake_k_from_image)
        fake_layout.addRow(self._fake_k_btn)
        fake_group.setLayout(fake_layout)
        layout.addWidget(fake_group)
        result_group = QGroupBox("Last calibration (K & distortion)")
        self._result_text = QPlainTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setMaximumBlockCount(500)
        self._result_text.setPlaceholderText("Run calibration or load from saved settings.")
        result_group_layout = QVBoxLayout()
        result_group_layout.addWidget(self._result_text)
        result_group.setLayout(result_group_layout)
        layout.addWidget(result_group)
        layout.addStretch()

    def refresh_calibration_display(self) -> None:
        """Update the K & distortion display from state.calibration."""
        cal = self._state.calibration
        if cal is None:
            self._result_text.setPlainText("")
            return
        lines = ["K (3×3):"]
        for row in cal.K:
            lines.append("  " + "  ".join(f"{x:.4f}" for x in row))
        d = cal.dist.ravel()
        lines.append("Distortion (k1, k2, p1, p2, k3, ...):")
        lines.append("  " + "  ".join(f"{x:.6f}" for x in d[: min(8, len(d))]))
        lines.append("Image size:")
        lines.append(f"  {cal.image_size[0]}×{cal.image_size[1]}" if cal.image_size else "  ...")
        if cal.reproj_err != 0.0 or cal.image_paths:
            lines.append(f"Reprojection error: {cal.reproj_err:.6f}")
            if cal.image_paths:
                lines.append(f"Images used: {len(cal.image_paths)}")
        self._result_text.setPlainText("\n".join(lines))

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
        self._state.calibration_is_fake = False
        self._state.calibration = result
        log_calibration_result(result)
        self.refresh_calibration_display()
        self.calibration_done.emit()

    def _on_fake_k_from_image(self) -> None:
        """Generate fake K from current center image; overwrites K in memory (not persisted)."""
        if self._get_current_image_size is None:
            logging_ui.log("Fake K: no access to current image (internal error).")
            return
        size = self._get_current_image_size()
        if size is None:
            logging_ui.log("Fake K: open an image in the center first.")
            return
        w, h = size
        h_fov = self._fake_fov_spin.value()
        K = fake_K_from_image_size(w, h, h_fov)
        dist = np.zeros(5, dtype=np.float64)
        self._state.calibration_is_fake = True
        self._state.calibration = CalibrationResult(
            K=K,
            dist=dist,
            rvecs=[],
            tvecs=[],
            reproj_err=0.0,
            image_paths=[],
            image_size=(w, h),
        )
        logging_ui.log(f"Fake K set from image {w}×{h}, horizontal FOV {h_fov}° (not saved).")
        self.refresh_calibration_display()
        self.calibration_done.emit()
