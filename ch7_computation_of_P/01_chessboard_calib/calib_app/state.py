"""
Application state for chessboard calibration: gallery paths, calibration results.

Single responsibility: hold gallery image paths, chessboard params, and calibration output (K, dist, R, t per image).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ChessboardParams:
    """Chessboard grid size (inner corners) and square size in world units."""

    cols: int = 8
    rows: int = 5
    square_size: float = 1.0  # e.g. 1.0 for unit squares


@dataclass
class CalibrationResult:
    """Result of calibrateCamera: K, distortion, per-image R/t, errors."""

    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (5,) or (N,)
    rvecs: List[np.ndarray]  # list of (3, 1)
    tvecs: List[np.ndarray]  # list of (3, 1)
    reproj_err: float
    image_paths: List[Path] = field(default_factory=list)


class AppState:
    """Mutable app state: gallery folder, image list, chessboard params, calibration result."""

    def __init__(self, gallery_folder: Path | None = None):
        if gallery_folder is None:
            gallery_folder = Path(__file__).resolve().parent.parent / "images" / "calib"
        self.gallery_folder = Path(gallery_folder)
        self.gallery_folder.mkdir(parents=True, exist_ok=True)
        self.image_paths: List[Path] = []
        self.chessboard = ChessboardParams()
        self.calibration: CalibrationResult | None = None
        self.calibration_is_fake: bool = False  # if True, do not persist K/dist
        self.selected_image_path: Path | None = None  # for main view
        self.current_demo_id: str = "none"
        self.realtime_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (R_cam_to_world, t_cam_to_world)
        self.realtime_display_frame: Optional[np.ndarray] = None  # BGR frame to show in central view (realtime pose demo)
        self.latest_camera_frame: Optional[np.ndarray] = None  # latest BGR frame for gallery capture

    def refresh_gallery_paths(self) -> List[Path]:
        """Scan gallery folder for images; update self.image_paths; return list."""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        paths = []
        for p in sorted(self.gallery_folder.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
        self.image_paths = paths
        return paths
