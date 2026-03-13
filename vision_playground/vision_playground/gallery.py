"""
Gallery: reusable dockable thumbnail grid for a folder; optional capture button.

Single responsibility: display and manage a folder-based image gallery.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

import cv2
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QGridLayout,
    QSizePolicy,
)


THUMB_SIZE = 120
# Extra horizontal space per tile (margins + spacing) for column count
THUMB_CELL_WIDTH = THUMB_SIZE + 16


def open_folder_in_explorer(path: Path) -> None:
    """Open the given folder in the system file manager."""
    path = Path(path).resolve()
    if not path.is_dir():
        return
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        pass


def load_thumbnail(path: Path, size: int = THUMB_SIZE) -> Optional[QPixmap]:
    """Load image and return scaled pixmap for thumbnail."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ThumbnailTile(QFrame):
    """Single thumbnail with optional remove button; click to select."""

    clicked = pyqtSignal(Path)

    def __init__(self, path: Path, on_remove: Optional[Callable[[Path], None]] = None, parent=None):
        super().__init__(parent)
        self.path = path
        self._on_remove = on_remove
        self.setFrameStyle(QFrame.StyledPanel)
        self.setLineWidth(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._label = QLabel()
        self._label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pix = load_thumbnail(path)
        if pix is not None:
            self._label.setPixmap(pix.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self._label.setText("?")
        layout.addWidget(self._label)
        name_label = QLabel(path.name)
        name_label.setWordWrap(True)
        name_label.setMaximumWidth(THUMB_SIZE)
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        rm_btn = QPushButton("Remove")
        rm_btn.setMaximumWidth(THUMB_SIZE)
        rm_btn.clicked.connect(self._do_remove)
        layout.addWidget(rm_btn)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(event)

    def _do_remove(self) -> None:
        if self._on_remove is not None:
            self._on_remove(self.path)


class GalleryWidget(QWidget):
    """Reusable gallery: thumbnails for a folder, optional capture button. Click thumbnail to open in center."""

    image_selected = pyqtSignal(Path)

    def __init__(
        self,
        folder: Path,
        title: str = "Gallery",
        on_capture_requested: Optional[Callable[[Path], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._title = title
        self._on_capture_requested = on_capture_requested
        self._tiles: List[ThumbnailTile] = []
        self._init_ui()
        self.reload()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel(self._title))
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.reload)
        header.addWidget(reload_btn)
        self._capture_btn = None
        if self._on_capture_requested is not None:
            self._capture_btn = QPushButton("Capture")
            self._capture_btn.setEnabled(False)
            self._capture_btn.clicked.connect(self._request_capture)
            header.addWidget(self._capture_btn)
        open_folder_btn = QPushButton("Open folder")
        open_folder_btn.clicked.connect(self._open_folder)
        header.addWidget(open_folder_btn)
        header.addStretch()
        layout.addLayout(header)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(4)
        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll)

    def set_capture_enabled(self, enabled: bool) -> None:
        """Enable or disable the Capture button (e.g. when preview is on/off)."""
        if self._capture_btn is not None:
            self._capture_btn.setEnabled(enabled)

    def _request_capture(self) -> None:
        """Ask main window to save current camera frame to this gallery's folder."""
        if self._on_capture_requested is not None:
            self._on_capture_requested(self._folder)

    def _open_folder(self) -> None:
        """Open the gallery folder in the system file manager."""
        open_folder_in_explorer(self._folder)

    def _clear_tiles(self) -> None:
        for t in self._tiles:
            t.deleteLater()
        self._tiles = []

    def _cols_from_width(self, width: int) -> int:
        """Number of columns that fit in the given width (Explorer-like auto-fit)."""
        if width <= 0:
            return 1
        return max(1, width // THUMB_CELL_WIDTH)

    def _relayout(self) -> None:
        """Re-arrange tiles in the grid to fit current viewport width."""
        viewport_width = self._scroll.viewport().width()
        cols = self._cols_from_width(viewport_width)
        for i, tile in enumerate(self._tiles):
            self._grid_layout.removeWidget(tile)
            row, col = i // cols, i % cols
            self._grid_layout.addWidget(tile, row, col)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._relayout()

    def reload(self) -> None:
        """Scan folder and rebuild thumbnail grid."""
        self._clear_tiles()
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = sorted(
            p for p in self._folder.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        for path in paths:
            tile = ThumbnailTile(path, on_remove=self._remove_image)
            tile.clicked.connect(self.image_selected.emit)
            self._tiles.append(tile)
        self._relayout()

    def _remove_image(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass
        self.reload()

    def get_paths(self) -> List[Path]:
        """Return current list of image paths in gallery."""
        return [t.path for t in self._tiles]
