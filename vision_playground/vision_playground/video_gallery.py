"""
Video gallery: dockable thumbnail grid for video files in a folder.
Thumbnails are generated from the first frame and cached on disk for speed.
"""
from __future__ import annotations

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
THUMB_CELL_WIDTH = THUMB_SIZE + 16
VIDEO_EXTENSIONS = {".mp4"}
THUMBNAIL_CACHE_DIR = ".thumbnails"


def _thumbnail_cache_path(video_folder: Path, video_path: Path) -> Path:
    """Path for cached thumbnail image (first frame) for a video."""
    cache_dir = video_folder / THUMBNAIL_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{video_path.stem}.jpg"


def load_video_thumbnail(video_path: Path, video_folder: Path, size: int = THUMB_SIZE) -> Optional[QPixmap]:
    """
    Load thumbnail for a video: use cached image if present, else extract first frame and cache.
    Returns QPixmap or None on failure.
    """
    cache_path = _thumbnail_cache_path(video_folder, video_path)
    if cache_path.exists():
        img = cv2.imread(str(cache_path))
        if img is not None:
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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        cv2.imwrite(str(cache_path), frame)
    except OSError:
        pass
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class VideoThumbnailTile(QFrame):
    """Single video thumbnail; click to open in center."""

    clicked = pyqtSignal(Path)

    def __init__(self, path: Path, video_folder: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self._video_folder = video_folder
        self.setFrameStyle(QFrame.StyledPanel)
        self.setLineWidth(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self._label = QLabel()
        self._label.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        pix = load_video_thumbnail(path, video_folder)
        if pix is not None:
            self._label.setPixmap(
                pix.scaled(THUMB_SIZE, THUMB_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self._label.setText("?")
        layout.addWidget(self._label)
        name_label = QLabel(path.name)
        name_label.setWordWrap(True)
        name_label.setMaximumWidth(THUMB_SIZE)
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(event)


class VideoGalleryWidget(QWidget):
    """Reusable video gallery: thumbnails for a folder (.mp4). Click to open in center."""

    video_selected = pyqtSignal(Path)

    def __init__(self, folder: Path, title: str = "Videos", parent=None):
        super().__init__(parent)
        self._folder = Path(folder)
        self._folder.mkdir(parents=True, exist_ok=True)
        self._title = title
        self._tiles: List[VideoThumbnailTile] = []
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

    def _cols_from_width(self, width: int) -> int:
        if width <= 0:
            return 1
        return max(1, width // THUMB_CELL_WIDTH)

    def _relayout(self) -> None:
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
        """Scan folder for video files and rebuild thumbnail grid."""
        self._clear_tiles()
        paths = sorted(
            p for p in self._folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        for path in paths:
            tile = VideoThumbnailTile(path, self._folder, self)
            tile.clicked.connect(self.video_selected.emit)
            self._tiles.append(tile)
        self._relayout()

    def _clear_tiles(self) -> None:
        for t in self._tiles:
            t.deleteLater()
        self._tiles = []
