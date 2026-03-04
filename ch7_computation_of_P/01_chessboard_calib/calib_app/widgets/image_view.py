"""
Reusable image view widget: display image with optional homography, report mapped coords on hover.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt


class ImageViewWidget(QWidget):
    """
    Displays an image (BGR numpy array) with optional 3x3 homography.
    Emits hover_image_coords(x, y) or (None,) on mouse move/leave for status bar etc.
    """

    hover_image_coords = pyqtSignal(object)  # tuple[float, float] | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_bgr: Optional[np.ndarray] = None
        self._H: Optional[np.ndarray] = None
        self._img_wh: Optional[tuple[int, int]] = None
        self._pixmap_wh: Optional[tuple[int, int]] = None
        self._placeholder_text = "No image"
        layout = QVBoxLayout(self)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setMinimumSize(400, 300)
        self._label.setStyleSheet("background-color: #f0f0f0; color: #444;")
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._label)
        self.setMouseTracking(True)
        self._label.setMouseTracking(True)
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_image(self, bgr: np.ndarray | None, placeholder: str = "No image") -> None:
        self._img_bgr = bgr if bgr is None or bgr.size == 0 else np.asarray(bgr)
        self._placeholder_text = placeholder
        self._update_display()

    def set_homography(self, H: np.ndarray | None) -> None:
        self._H = np.asarray(H, dtype=np.float64) if H is not None else None
        self._update_display()

    def map_to_image_coords(self, wx: float, wy: float) -> tuple[float, float] | None:
        """Map widget coords to image pixel (x, y); None if outside image."""
        if self._img_wh is None or self._pixmap_wh is None:
            return None
        iw, ih = self._img_wh
        nw, nh = self._pixmap_wh
        ww, wh = self.width(), self.height()
        left = (ww - nw) / 2
        top = (wh - nh) / 2
        px, py = wx - left, wy - top
        if 0 <= px < nw and 0 <= py < nh:
            return (px * iw / nw, py * ih / nh)
        return None

    def _update_display(self) -> None:
        if self._img_bgr is None or self._img_bgr.size == 0:
            self._label.setText(getattr(self, "_placeholder_text", "No image"))
            self._label.setPixmap(QPixmap())
            self._img_wh = None
            self._pixmap_wh = None
            return
        img = self._img_bgr
        if self._H is not None:
            h, w = img.shape[:2]
            img = cv2.warpPerspective(img, self._H, (w, h))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        self._img_wh = (w, h)
        lw, lh = self._label.width(), self._label.height()
        if lw > 0 and lh > 0:
            scale = min(lw / w, lh / h, 1.0)
            if scale < 1.0:
                nw, nh = int(w * scale), int(h * scale)
                if nw < 1:
                    nw = 1
                if nh < 1:
                    nh = 1
                img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                h, w = nh, nw
        self._pixmap_wh = (w, h)
        bytes_per_line = 3 * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._label.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._label.setText("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()

    def _emit_hover(self, wx: float, wy: float) -> None:
        pt = self.map_to_image_coords(wx, wy)
        self.hover_image_coords.emit(pt)

    def mouseMoveEvent(self, event) -> None:
        self._emit_hover(event.pos().x(), event.pos().y())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self.hover_image_coords.emit(None)
        super().leaveEvent(event)
