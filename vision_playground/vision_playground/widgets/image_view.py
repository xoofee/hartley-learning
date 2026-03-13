"""
Reusable image view widget: display image with optional homography, zoom (wheel), pan (right-drag), report mapped coords on hover.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QPushButton


class _PaintArea(QWidget):
    """Inner widget that draws the image with zoom/pan and handles wheel + right-drag."""

    def __init__(self, view: "ImageViewWidget", parent=None):
        super().__init__(parent)
        self._view = view
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #f0f0f0; color: #444;")
        self.setMouseTracking(True)
        self._panning = False
        self._last_pan_pos: Optional[tuple[float, float]] = None

    def _view_size(self) -> tuple[int, int]:
        return self.width(), self.height()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        v = self._view
        if v._display_rgb is None or v._display_rgb.size == 0:
            painter.drawText(self.rect(), Qt.AlignCenter, getattr(v, "_placeholder_text", "No image"))
            return
        img_h, img_w = v._display_rgb.shape[:2]
        ww, wh = self._view_size()
        if ww <= 0 or wh <= 0:
            return
        scale_x = ww / img_w
        scale_y = wh / img_h
        base_scale = min(scale_x, scale_y)
        scale = base_scale * v._zoom_scale
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        if scaled_w < 1 or scaled_h < 1:
            return
        x_offset = (ww - scaled_w) / 2 + v._pan_dx
        y_offset = (wh - scaled_h) / 2 + v._pan_dy
        qimg = QImage(
            v._display_rgb.data,
            img_w,
            img_h,
            img_w * 3,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg.copy()).scaled(
            scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(int(x_offset), int(y_offset), pixmap)

    def wheelEvent(self, event) -> None:
        v = self._view
        if v._display_rgb is None or v._display_rgb.size == 0:
            return
        mouse_x = event.x()
        mouse_y = event.y()
        img_x, img_y = v._widget_to_image_coords(mouse_x, mouse_y)
        if img_x is None:
            return
        zoom_delta = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = v._zoom_scale * zoom_delta
        new_zoom = max(0.1, min(20.0, new_zoom))
        if new_zoom == v._zoom_scale:
            return
        img_h, img_w = v._display_rgb.shape[:2]
        ww, wh = self._view_size()
        scale_x = ww / img_w
        scale_y = wh / img_h
        base_scale = min(scale_x, scale_y)
        old_scale = base_scale * v._zoom_scale
        new_scale = base_scale * new_zoom
        old_x_offset = (ww - img_w * old_scale) / 2 + v._pan_dx
        old_y_offset = (wh - img_h * old_scale) / 2 + v._pan_dy
        new_x_offset = mouse_x - img_x * new_scale
        new_y_offset = mouse_y - img_y * new_scale
        base_x_offset = (ww - img_w * new_scale) / 2
        base_y_offset = (wh - img_h * new_scale) / 2
        v._pan_dx = new_x_offset - base_x_offset
        v._pan_dy = new_y_offset - base_y_offset
        v._zoom_scale = new_zoom
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._panning = True
            self._last_pan_pos = (event.x(), event.y())
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._last_pan_pos is not None:
            x, y = event.x(), event.y()
            dx = x - self._last_pan_pos[0]
            dy = y - self._last_pan_pos[1]
            self._view._pan_dx += dx
            self._view._pan_dy += dy
            self._last_pan_pos = (x, y)
            self.update()
        else:
            self._view._emit_hover_from_area(event.x(), event.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            self._panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._view.hover_image_coords.emit(None)
        super().leaveEvent(event)


class ImageViewWidget(QWidget):
    """
    Displays an image (BGR numpy array) with optional 3x3 homography.
    Zoom: mouse wheel. Pan: right-button drag. Fit button resets zoom/pan.
    Emits hover_image_coords(x, y) or (None,) on mouse move/leave for status bar etc.
    """

    hover_image_coords = pyqtSignal(object)  # tuple[float, float] | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_bgr: Optional[np.ndarray] = None
        self._H: Optional[np.ndarray] = None
        self._display_rgb: Optional[np.ndarray] = None
        self._img_wh: Optional[tuple[int, int]] = None
        self._pixmap_wh: Optional[tuple[int, int]] = None
        self._placeholder_text = "No image"
        self._zoom_scale = 1.0
        self._pan_dx = 0.0
        self._pan_dy = 0.0
        layout = QVBoxLayout(self)
        self._paint_area = _PaintArea(self, self)
        layout.addWidget(self._paint_area)
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setToolTip("Reset zoom and pan to fit image")
        self._fit_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._fit_btn.clicked.connect(self._on_fit)
        layout.addWidget(self._fit_btn, 0, Qt.AlignLeft)

    def _on_fit(self) -> None:
        self._zoom_scale = 1.0
        self._pan_dx = 0.0
        self._pan_dy = 0.0
        self._paint_area.update()

    def set_image(self, bgr: np.ndarray | None, placeholder: str = "No image", preserve_view: bool = False) -> None:
        """Set the image to display. If preserve_view is True, zoom and pan are not reset (e.g. when only overlay changes)."""
        self._img_bgr = bgr if bgr is None or bgr.size == 0 else np.asarray(bgr)
        self._placeholder_text = placeholder
        if not preserve_view:
            self._zoom_scale = 1.0
            self._pan_dx = 0.0
            self._pan_dy = 0.0
        self._update_display()

    def set_homography(self, H: np.ndarray | None) -> None:
        self._H = np.asarray(H, dtype=np.float64) if H is not None else None
        self._update_display()

    def _widget_to_image_coords(self, wx: float, wy: float) -> Optional[tuple[float, float]]:
        """Map paint-area coords (wx, wy) to image coords; None if outside image."""
        if self._display_rgb is None or self._img_wh is None:
            return None
        iw, ih = self._img_wh
        img_h, img_w = self._display_rgb.shape[:2]
        ww, wh = self._paint_area.width(), self._paint_area.height()
        if ww <= 0 or wh <= 0:
            return None
        scale_x = ww / img_w
        scale_y = wh / img_h
        base_scale = min(scale_x, scale_y)
        scale = base_scale * self._zoom_scale
        scaled_w = img_w * scale
        scaled_h = img_h * scale
        left = (ww - scaled_w) / 2 + self._pan_dx
        top = (wh - scaled_h) / 2 + self._pan_dy
        px, py = wx - left, wy - top
        if 0 <= px < scaled_w and 0 <= py < scaled_h:
            return (px * iw / scaled_w, py * ih / scaled_h)
        return None

    def map_to_image_coords(self, wx: float, wy: float) -> tuple[float, float] | None:
        """Map widget coords (in this widget's space) to image pixel (x, y); None if outside image."""
        r = self._paint_area.geometry()
        if not r.contains(int(wx), int(wy)):
            return None
        return self._widget_to_image_coords(wx - r.x(), wy - r.y())

    def _emit_hover_from_area(self, wx: float, wy: float) -> None:
        pt = self._widget_to_image_coords(wx, wy)
        self.hover_image_coords.emit(pt)

    def _update_display(self) -> None:
        if self._img_bgr is None or self._img_bgr.size == 0:
            self._display_rgb = None
            self._img_wh = None
            self._pixmap_wh = None
            self._paint_area.update()
            return
        img = self._img_bgr
        if self._H is not None:
            h, w = img.shape[:2]
            img = cv2.warpPerspective(img, self._H, (w, h))
        self._display_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = self._display_rgb.shape[:2]
        self._img_wh = (w, h)
        lw = self._paint_area.width()
        lh = self._paint_area.height()
        if lw > 0 and lh > 0:
            scale = min(lw / w, lh / h, 1.0)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            self._pixmap_wh = (nw, nh)
        else:
            self._pixmap_wh = (w, h)
        self._paint_area.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()
