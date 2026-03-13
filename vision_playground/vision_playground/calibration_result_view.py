"""
Calibration result images view: left column of thumbnails, right main area shows
selected image with drawChessboardCorners and first corner highlighted by a red circle.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QSizePolicy,
)

from .state import AppState, CalibrationResult, ChessboardParams
from .calibration import find_chessboard_corners

THUMB_SIZE = 100
FIRST_CORNER_CIRCLE_RADIUS = 25  # bigger than drawChessboardCorners default


def _load_thumbnail(path: Path, size: int = THUMB_SIZE) -> Optional[QPixmap]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1:
            nw = 1
        if nh < 1:
            nh = 1
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = nh, nw
    bytes_per_line = 3 * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def _build_result_image(
    path: Path,
    cols: int,
    rows: int,
    first_corner_radius: int = FIRST_CORNER_CIRCLE_RADIUS,
) -> Optional[np.ndarray]:
    """Load image, find corners, drawChessboardCorners, draw red circle on first corner. Returns BGR."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    corners, found = find_chessboard_corners(img, cols, rows)
    if not found or corners is None:
        return img
    img = img.copy()
    cv2.drawChessboardCorners(img, (cols, rows), corners, found)
    # First corner: corners[0] shape (1, 2)
    x, y = int(corners[0, 0, 0]), int(corners[0, 0, 1])
    cv2.circle(img, (x, y), first_corner_radius, (0, 0, 255), 2)
    return img


class CalibrationResultImagesWidget(QWidget):
    """
    Left: column of thumbnails (one per row).
    Right: main area showing selected image with corners and first corner highlighted.
    """

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self._state = state
        self._paths: List[Path] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Left: list of thumbnails
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ListMode)
        self._list.setIconSize(QSize(THUMB_SIZE, THUMB_SIZE))
        self._list.setMaximumWidth(THUMB_SIZE + 40)
        self._list.setMinimumWidth(THUMB_SIZE + 24)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        # Right: main image (fit to widget, aspect ratio preserved)
        self._main_label = QLabel()
        self._main_label.setAlignment(Qt.AlignCenter)
        self._main_label.setMinimumSize(400, 300)
        self._main_label.setStyleSheet("background-color: #e8e8e8; color: #555;")
        self._main_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._main_label.setScaledContents(False)
        self._current_img_bgr: Optional[np.ndarray] = None
        layout.addWidget(self._main_label, 1)

    def set_calibration_result(self, result: Optional[CalibrationResult], chessboard: ChessboardParams) -> None:
        """Populate from calibration result; show first image by default."""
        self._paths = list(result.image_paths) if result and result.image_paths else []
        self._list.clear()
        if not self._paths:
            self._main_label.setText("No calibration images.\nRun calibration to see result images.")
            self._main_label.setPixmap(QPixmap())
            return
        for path in self._paths:
            pix = _load_thumbnail(path)
            item = QListWidgetItem(path.name)
            if pix is not None:
                item.setIcon(QIcon(pix))
            item.setData(Qt.UserRole, path)
            self._list.addItem(item)
        self._list.setCurrentRow(0)  # triggers currentRowChanged -> _show_image_at_index(0)

    def _on_selection_changed(self, row: int) -> None:
        if row >= 0:
            self._show_image_at_index(row)

    def _show_image_at_index(self, index: int) -> None:
        if index < 0 or index >= len(self._paths):
            return
        path = self._paths[index]
        cb = self._state.chessboard
        img_bgr = _build_result_image(path, cb.cols, cb.rows)
        if img_bgr is None:
            self._current_img_bgr = None
            self._main_label.setText(f"Failed to load: {path.name}")
            self._main_label.setPixmap(QPixmap())
            return
        self._current_img_bgr = img_bgr
        self._set_main_pixmap_from_bgr()

    def _set_main_pixmap_from_bgr(self) -> None:
        """Scale current image to fit the label and set pixmap (aspect ratio preserved)."""
        if self._current_img_bgr is None or self._current_img_bgr.size == 0:
            return
        img = self._current_img_bgr
        if len(img.shape) == 2:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        if w <= 0 or h <= 0:
            return
        label_w = self._main_label.width()
        label_h = self._main_label.height()
        scale = min(label_w / w, label_h / h, 1.0)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            if nw < 1:
                nw = 1
            if nh < 1:
                nh = 1
            frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = nh, nw
        bytes_per_line = 3 * w
        qimg = QImage(
            frame_rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self._main_label.setPixmap(QPixmap.fromImage(qimg))
        self._main_label.setText("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._current_img_bgr is not None:
            self._set_main_pixmap_from_bgr()
