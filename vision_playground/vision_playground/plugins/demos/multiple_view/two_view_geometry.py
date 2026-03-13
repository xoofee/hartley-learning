"""
Two-view geometry: F/E estimation, pose recovery, triangulation, reprojection error, epilines.
Pure functions, no Qt. Used by two_view_reconstruction demo.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def estimate_F(pts1: np.ndarray, pts2: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC fundamental matrix. Returns F (3,3) or None, mask (N,)."""
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
    )
    if F is None or mask is None:
        return None, np.zeros(pts1.shape[0], dtype=np.uint8)
    return F, mask.ravel()


def estimate_E(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray]:
    """RANSAC essential matrix. Returns E (3,3) or None, mask (N,)."""
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
    if E is None or mask is None:
        return None, np.zeros(pts1.shape[0], dtype=np.uint8)
    return E, mask.ravel()


def recover_pose(
    E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Recover R, t from E. Returns R (3,3), t (3,1), mask (N,). Convention: P2 = K[R|t], so cam2 center in world C2 = -R.T @ t."""
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=None)
    return R, t, mask.ravel()


def P2_from_F(F: np.ndarray) -> np.ndarray:
    """Camera matrix P2 (3,4) from F with P1 = [I|0]. P2 = [[e']_x F | e']."""
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]
    e2 = e2 / (e2[2] if e2[2] != 0 else 1e-10)
    ex = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    return np.hstack([ex @ F, e2.reshape(3, 1)])


def projection_matrices(
    K: Optional[np.ndarray],
    R: Optional[np.ndarray],
    t: Optional[np.ndarray],
    F: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return P1 (3,4), P2 (3,4). If K known: P1 = K[I|0], P2 = K[R|t]. Else: P1 = [I|0], P2 from F."""
    if K is not None and R is not None and t is not None:
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        return P1, P2
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
    if F is not None:
        P2 = P2_from_F(F)
    else:
        P2 = np.hstack([np.eye(3), np.ones((3, 1))])
    return P1, P2


def triangulate(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Triangulate inliers. pts1/pts2 (N,2), mask (N,). Returns pts3d (3, M)."""
    in1 = pts1[mask == 1]
    in2 = pts2[mask == 1]
    if in1.shape[0] == 0:
        return np.zeros((3, 0))
    pts4d = cv2.triangulatePoints(P1, P2, in1.T, in2.T)
    return pts4d[:3] / (pts4d[3] + 1e-10)


def point_colors(img: np.ndarray, pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sample BGR at pts (N,2); return (M,3) in [0,1] for inliers (RGB for matplotlib)."""
    in_pts = pts[mask == 1]
    if in_pts.shape[0] == 0:
        return np.zeros((0, 3))
    h, w = img.shape[:2]
    x = np.clip(in_pts[:, 0].astype(int), 0, w - 1)
    y = np.clip(in_pts[:, 1].astype(int), 0, h - 1)
    colors = img[y, x]
    return colors[:, ::-1] / 255.0


def camera_poses_from_Rt(
    R: np.ndarray, t: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """From P2 = K[R|t], return (R1, t1), (R2, t2) as R_cam_to_world, t_cam_to_world for plot3d."""
    t = t.ravel()
    R1 = np.eye(3)
    t1 = np.zeros(3)
    C2 = -R.T @ t
    R2 = R.T
    return (R1, t1), (R2, C2)


def fundamental_from_E(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """F = K^{-T} E K^{-1} for epipolar line visualization when using calibrated E."""
    Kinv = np.linalg.inv(K)
    return np.linalg.inv(K).T @ E @ Kinv


def reprojection_error(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1_in: np.ndarray,
    pts2_in: np.ndarray,
    pts3d: np.ndarray,
) -> np.ndarray:
    """Per-point mean reprojection error (pixels). pts1_in, pts2_in (M,2); pts3d (3,M). Returns (M,) errors."""
    X_h = np.vstack([pts3d, np.ones((1, pts3d.shape[1]))])
    x1 = (P1 @ X_h)[:2] / (P1 @ X_h)[2]
    x2 = (P2 @ X_h)[:2] / (P2 @ X_h)[2]
    err1 = np.linalg.norm(x1.T - pts1_in, axis=1)
    err2 = np.linalg.norm(x2.T - pts2_in, axis=1)
    return (err1 + err2) / 2.0


def compute_epilines(pts: np.ndarray, F: np.ndarray, which_image: int) -> np.ndarray:
    """Lines in the other image. pts (N,2), which_image 1 or 2. Returns (N,3) lines ax+by+c=0."""
    pts_ = pts.reshape(-1, 1, 2).astype(np.float32)
    lines = cv2.computeCorrespondEpilines(pts_, which_image, F)
    return lines.reshape(-1, 3)


def default_K_for_pyramid() -> np.ndarray:
    """Default K for drawing camera pyramids when uncalibrated."""
    return np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]], dtype=np.float64)
