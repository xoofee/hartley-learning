"""
P from 6 points: normalized (sqrt(3)) DLT to compute P from 6 point correspondences.

4 points on the ground (z=0) in a rectangle, 2 points in the air. Scene shapes are hidden.
Shows the 6 points in different colors with labels 1–6; logs DLT vs original P error.
"""
from __future__ import annotations

import numpy as np

from ..registry import Demo
from ... import pinhole
from ...logging_ui import log


# 6 world points: 4 on ground z=0 (rectangle), 2 in the air
# Rectangle corners and two elevated points
def _get_six_points() -> np.ndarray:
    """Return (6, 3) world points: 4 ground rectangle, 2 in the air."""
    # Ground rectangle (z=0): e.g. corners
    h = 0.6
    ground = np.array([
        [-h, -h, 0.0],
        [h, -h, 0.0],
        [h, h, 0.0],
        [-h, h, 0.0],
    ])
    # Two points in the air
    # air = np.array([
    #     [0.0, 0.0, 0.8],
    #     [0.3, 0.2, 0.5],
    # ])
    air = np.array([
        [-h, -h, 0.8],
        [h, -h, 0.8],
    ])
    return np.vstack([ground, air])


def _normalized_dlt_P(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    scale_3d: float = np.sqrt(3),
    scale_2d: float = np.sqrt(2),
) -> np.ndarray:
    """
    Compute 3x4 camera matrix P from 3D–2D point correspondences using normalized DLT.

    points_3d: (N, 3), N >= 6
    points_2d: (N, 2) image (u, v)
    scale_3d: target scale for 3D (average distance from centroid after centering)
    scale_2d: target scale for 2D (average distance from centroid after centering)
    Returns: P (3, 4) such that x ~ P @ [X; 1].
    """
    n = points_3d.shape[0]
    assert n >= 6 and points_2d.shape[0] == n

    # 3D normalization: centroid to origin, scale so mean distance = scale_3d
    c3 = np.mean(points_3d, axis=0)
    X_centered = points_3d - c3
    mean_dist_3d = np.mean(np.linalg.norm(X_centered, axis=1))
    if mean_dist_3d < 1e-10:
        mean_dist_3d = 1e-10
    s3 = scale_3d / mean_dist_3d
    T3 = np.eye(4)
    T3[:3, :3] = s3 * np.eye(3)
    T3[:3, 3] = -s3 * c3
    X_hom = np.hstack([points_3d, np.ones((n, 1))])
    X_norm = (T3 @ X_hom.T).T  # (n, 4)

    # 2D normalization: centroid to origin, scale so mean distance = scale_2d
    t2 = np.mean(points_2d, axis=0)
    x_centered = points_2d - t2
    mean_dist_2d = np.mean(np.linalg.norm(x_centered, axis=1))
    if mean_dist_2d < 1e-10:
        mean_dist_2d = 1e-10
    s2 = scale_2d / mean_dist_2d
    T2 = np.eye(3)
    T2[0, 0] = T2[1, 1] = s2
    T2[0, 2] = -s2 * t2[0]
    T2[1, 2] = -s2 * t2[1]
    x_hom = np.hstack([points_2d, np.ones((n, 1))])
    x_norm = (T2 @ x_hom.T).T  # (n, 3)

    # DLT: x_norm × (P_norm @ X_norm) = 0  =>  two rows per point
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X = X_norm[i]
        u, v, w = x_norm[i, 0], x_norm[i, 1], x_norm[i, 2]
        # Row 1: -w*(p2·X) + v*(p3·X) = 0
        A[2 * i, 4:8] = -w * X
        A[2 * i, 8:12] = v * X
        # Row 2: w*(p1·X) - u*(p3·X) = 0
        A[2 * i + 1, 0:4] = w * X
        A[2 * i + 1, 8:12] = -u * X

    _, _, Vt = np.linalg.svd(A)
    p_flat = Vt[-1]
    P_norm = p_flat.reshape(3, 4)

    # Denormalize: x = T2^{-1} @ P_norm @ T3 @ X  =>  P = T2^{-1} @ P_norm @ T3
    T2_inv = np.linalg.inv(T2)
    P = T2_inv @ P_norm @ T3
    return P


def _error_to_original_P(P_est: np.ndarray, P_orig: np.ndarray) -> tuple[float, float]:
    """
    Scale P_est to match P_orig and return Frobenius difference and reprojection RMSE.
    Returns (frobenius_diff, reproj_rmse).
    """
    # Scale P_est so that Frobenius norm matches P_orig (or use last element)
    # scale = np.linalg.norm(P_orig, "fro") / (np.linalg.norm(P_est, "fro") + 1e-12)
    scale = P_orig[2, 3] / P_est[2, 3]
    P_est_scaled = P_est * scale
    frob = float(np.linalg.norm(P_orig - P_est_scaled, "fro"))
    return frob, 0.0  # reproj computed separately if needed


class PFrom6PointsDemo(Demo):
    """
    Compute P from 6 point correspondences using normalized (sqrt(3)) DLT.
    Hides other shapes; shows 6 points with colors and labels 1–6; logs error to original P.
    """

    # Distinct colors for points 1–6
    COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
    LABELS = ["1", "2", "3", "4", "5", "6"]

    def __init__(self) -> None:
        self._points_3d = _get_six_points()
        self._last_log_P_hash: int | None = None

    def id(self) -> str:
        return "p_from_6_points"

    def label(self) -> str:
        return "P from 6 points"

    def hide_scene_shapes(self) -> bool:
        return True

    def on_activated(self, context: dict) -> None:
        pass

    def on_deactivated(self) -> None:
        pass

    def on_draw_3d(self, ax3d, context: dict) -> None:
        for i in range(6):
            pt = self._points_3d[i]
            ax3d.scatter(
                [pt[0]], [pt[1]], [pt[2]],
                c=self.COLORS[i], s=80, zorder=5, edgecolors="white", linewidths=1.5,
            )
            ax3d.text(pt[0], pt[1], pt[2], f"  {self.LABELS[i]}", fontsize=10, color="white")

    def on_draw_image(self, ax_img, context: dict) -> None:
        state = context.get("state")
        P = context.get("P")
        if state is None or P is None:
            return
        # Project 6 points with current P (ideal, no distortion for DLT input)
        points_2d = pinhole.project_points(P, self._points_3d)
        # Draw with colors and labels
        for i in range(6):
            u, v = points_2d[i, 0], points_2d[i, 1]
            ax_img.scatter(
                u, v, c=self.COLORS[i], s=100, zorder=10, edgecolors="white", linewidths=2,
            )
            ax_img.text(u, v, f"  {self.LABELS[i]}", fontsize=10, color="white", zorder=11)

        # Normalized DLT (sqrt(3) for 3D); log error to original P
        try:
            P_est = _normalized_dlt_P(
                self._points_3d,
                points_2d,
                scale_3d=np.sqrt(3),
                scale_2d=np.sqrt(2),
            )
            frob, _ = _error_to_original_P(P_est, P)
            pts_reproj = pinhole.project_points(P_est, self._points_3d)
            reproj_err = np.sqrt(np.mean((points_2d - pts_reproj) ** 2))
            # Log when P (or thus error) effectively changed to avoid flooding
            P_hash = hash(P.tobytes())
            if P_hash != self._last_log_P_hash:
                self._last_log_P_hash = P_hash
                log(
                    f"P from 6 pts: Frobenius |P_orig - P_dlt| = {frob:.6g}; "
                    f"reproj RMSE (px) = {reproj_err:.6g}"
                )
                log(f"P =\n{np.array2string(P, precision=6, suppress_small=True)}")
                log(f"P_dlt =\n{np.array2string(P_est, precision=6, suppress_small=True)}")
        except Exception as e:
            log(f"P from 6 pts DLT error: {e}")
