"""
Panoramic mosaic demo: use open images in center widget, sort by name, middle as reference,
find H between consecutive images, warp into one canvas (simple imprint, no fusion).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QPushButton

from ..registry import Demo


def _find_H_from_pair(img_left: np.ndarray, img_right: np.ndarray) -> Optional[np.ndarray]:
    """Find homography H such that right ≈ H @ left (maps left image coords to right). Returns 3x3 or None."""
    orb = cv2.ORB_create(nfeatures=2000)
    gleft = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gright = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    kp1, d1 = orb.detectAndCompute(gleft, None)
    kp2, d2 = orb.detectAndCompute(gright, None)
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    if len(matches) < 4:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[: min(500, len(matches))]
    pts_left = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_left, pts_right, cv2.RANSAC, 5.0)
    return H


def _build_mosaic(
    images: list[tuple[Path, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    images: list of (path, bgr) sorted by path name.
    Use middle image as reference. Find H between consecutive pairs, compose to ref, then warp all into one canvas.
    Returns BGR canvas or None on failure.
    """
    n = len(images)
    if n < 2:
        return None
    ref_idx = n // 2

    # H[i] = homography from image i to image i+1 (so coords in i+1 = H[i] @ coords in i)
    H_consec: list[Optional[np.ndarray]] = [None] * (n - 1)
    for i in range(n - 1):
        _, img_i = images[i]
        _, img_i1 = images[i + 1]
        H_consec[i] = _find_H_from_pair(img_i, img_i1)
        if H_consec[i] is None:
            return None

    # H_i_to_ref: from image i coords to reference frame coords (point_ref = H_i_to_ref @ point_i)
    H_to_ref: list[Optional[np.ndarray]] = [None] * n
    H_to_ref[ref_idx] = np.eye(3, dtype=np.float64)
    for i in range(ref_idx - 1, -1, -1):
        # p_ref = H_consec[ref-1] @ ... @ H_consec[i] @ p_i => H_i_to_ref = H_to_ref[i+1] @ H_consec[i]
        H_to_ref[i] = H_to_ref[i + 1] @ H_consec[i]
    for i in range(ref_idx + 1, n):
        # p_ref = inv(H_ref_to_i) @ p_i, H_ref_to_i = H_consec[ref] @ ... @ H_consec[i-1]
        H_ref_to_i = np.eye(3, dtype=np.float64)
        for j in range(ref_idx, i):
            H_ref_to_i = H_ref_to_i @ H_consec[j]
        H_to_ref[i] = np.linalg.inv(H_ref_to_i)

    # Bounding box of all warped image corners
    h_ref, w_ref = images[ref_idx][1].shape[:2]
    corners_ref = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = [cv2.perspectiveTransform(corners_ref, H_to_ref[ref_idx])]
    for i in range(n):
        if i == ref_idx:
            continue
        h, w = images[i][1].shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H_to_ref[i])
        all_corners.append(warped)
    all_pts = np.vstack(all_corners).reshape(-1, 2)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    min_x, min_y = int(np.floor(min_xy[0])), int(np.floor(min_xy[1]))
    max_x, max_y = int(np.ceil(max_xy[0])), int(np.ceil(max_xy[1]))
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    if canvas_w <= 0 or canvas_h <= 0:
        return None
    # Translation to put (min_x, min_y) at (0, 0)
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = 0

    for i in range(n):
        H = T @ H_to_ref[i]
        img = images[i][1]
        h, w = img.shape[:2]
        warped = cv2.warpPerspective(img, H, (canvas_w, canvas_h))
        # Simple imprint: overwrite (no blending)
        mask = (warped[:, :, 0] != 0) | (warped[:, :, 1] != 0) | (warped[:, :, 2] != 0)
        canvas[mask] = warped[mask]

    return canvas


class PanoramicMosaicDemo(Demo):
    def id(self) -> str:
        return "panoramic_mosaic"

    def label(self) -> str:
        return "Panoramic mosaic"

    def on_activated(self, context: dict) -> None:
        self._context = context

    def on_deactivated(self) -> None:
        self._context = None

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        btn = QPushButton("Build mosaic")
        btn.setToolTip("Use open images (sorted by name), middle as reference, find H between consecutive, show single canvas.")
        btn.clicked.connect(lambda: self._run_mosaic(context))
        layout.addWidget(btn)
        layout.addStretch()
        return widget

    def _run_mosaic(self, context: dict) -> None:
        get_sorted = context.get("get_open_documents_sorted")
        open_path = context.get("open_path")
        get_work = context.get("get_work_folder")
        if not callable(get_sorted) or not callable(open_path) or not callable(get_work):
            QMessageBox.warning(None, "Panoramic mosaic", "Missing context (get_open_documents_sorted / open_path / get_work_folder).")
            return
        images = get_sorted()
        if len(images) < 2:
            QMessageBox.warning(None, "Panoramic mosaic", "Open at least 2 images in the center (from gallery).")
            return
        mosaic = _build_mosaic(images)
        if mosaic is None:
            QMessageBox.warning(None, "Panoramic mosaic", "Failed to find homographies (e.g. not enough matches between consecutive images).")
            return
        work = Path(get_work())
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / "mosaic_result.jpg"
        if cv2.imwrite(str(out_path), mosaic):
            open_path(out_path)
        else:
            QMessageBox.warning(None, "Panoramic mosaic", f"Could not save mosaic to {out_path}.")
