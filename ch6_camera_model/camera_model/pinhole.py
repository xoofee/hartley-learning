"""
Pinhole camera model: intrinsics K, rotation R, projection P, project/decompose.

Single responsibility: camera matrix math (K, R, t, P) and point projection.
"""
from __future__ import annotations

import numpy as np



def build_intrinsic(
    focal_length_mm: float,
    sensor_width_mm: float,
    sensor_height_mm: float,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
    cx_px: float | None = None,
    cy_px: float | None = None,
) -> np.ndarray:
    """
    Build 3x3 intrinsic matrix K (camera matrix).
    Units: focal_length_mm (mm), sensor (mm), pixel pitch (mm/pixel).
    Focal length in pixels: fx_px = focal_length_mm / pixel_size_x_mm, etc.
    """
    image_width_px = sensor_width_mm / pixel_size_x_mm
    image_height_px = sensor_height_mm / pixel_size_y_mm
    fx_px = focal_length_mm / pixel_size_x_mm
    fy_px = focal_length_mm / pixel_size_y_mm
    cx = cx_px if cx_px is not None else image_width_px / 2.0
    cy = cy_px if cy_px is not None else image_height_px / 2.0
    return np.array([
        [fx_px, 0, cx],
        [0, fy_px, cy],
        [0, 0, 1],
    ], dtype=np.float64)


def build_projection_matrix(
    K: np.ndarray,
    R_cw: np.ndarray,
    camera_center_world: np.ndarray,
) -> np.ndarray:
    """P = K [R_world_to_cam | t_world_to_cam]. R_wc = R_cam^T, t_wc = -R_wc @ camera_center_world. P is (3, 4)."""
    t_world_to_cam = (-R_cw @ camera_center_world).reshape(3, 1)
    Rt = np.hstack([R_cw, t_world_to_cam])
    return K @ Rt


def decompose_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose P (3,4) into K (3,3), R_world_to_cam (3,3), t_world_to_cam (3,1) with P = K [R_wc|t_wc]. Uses RQ on M = P[:,:3]."""
    from scipy.linalg import rq
    M = P[:, :3].copy()
    p = P[:, 3:4]
    K_intrinsic, R_world_to_cam = rq(M)
    D = np.diag(np.sign(np.diag(K_intrinsic)))
    D[np.diag(D) == 0] = 1
    K_intrinsic = K_intrinsic @ D
    R_world_to_cam = D.T @ R_world_to_cam
    if np.linalg.det(R_world_to_cam) < 0:
        R_world_to_cam = -R_world_to_cam
        K_intrinsic = -K_intrinsic
    t_world_to_cam = np.linalg.solve(K_intrinsic, p)
    return K_intrinsic, R_world_to_cam, t_world_to_cam


def project_points(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """points_world (N, 3) -> (N, 2) image coords (u, v) ideal (no distortion)."""
    n = points_world.shape[0]
    ones = np.ones((n, 1))
    hom = np.hstack([points_world, ones])
    x = (P @ hom.T).T
    w = x[:, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    u = x[:, 0] / w
    v = x[:, 1] / w
    return np.column_stack([u, v])


def project_one(P: np.ndarray, point_world: np.ndarray) -> tuple[float, float, float]:
    """Project one 3D point; returns (u, v, w). w is depth (camera z); w <= 0 means behind camera."""
    x = P @ np.append(point_world, 1.0)
    w = float(x[2])
    if abs(w) < 1e-12:
        w = 1e-12
    u = x[0] / w
    v = x[1] / w
    return u, v, w


def _affine_P(P: np.ndarray) -> np.ndarray:
    """
    Return 3x4 affine (telecentric) projection matrix by zeroing the last row of R,
    not the last row of P. So P_affine = K [R_affine | t] with R_affine[2,:] = 0.
    Then Z_cam = t[2] is constant and (cx, cy) no longer couple Z into (u,v).
    """
    K, R_world_to_cam, t_world_to_cam = decompose_P(P)
    R_affine = R_world_to_cam.copy()
    R_affine[2, :] = 0.0
    Rt_affine = np.hstack([R_affine, t_world_to_cam.reshape(3, 1)])

    P_affine = K @ Rt_affine

    print('P:', P)
    print('P_affine:', P_affine)

    return P_affine


def project_points_affine(P: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """
    Affine (telecentric) projection: Z_cam is constant (last row of R zeroed),
    so (u, v) = (x0/x2, x1/x2) with x2 = t[2] constant — linear in world coords.
    points_world (N, 3) -> (N, 2).
    """
    P_affine = _affine_P(P)
    n = points_world.shape[0]
    hom = np.hstack([points_world, np.ones((n, 1))])
    x = (P_affine @ hom.T).T
    w = x[:, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    u = x[:, 0] / w
    v = x[:, 1] / w

    return np.column_stack([u, v])


def project_one_affine(P: np.ndarray, point_world: np.ndarray) -> tuple[float, float, float]:
    """Affine projection for one point; returns (u, v, w) with w = t[2] constant."""
    P_affine = _affine_P(P)
    x = P_affine @ np.append(point_world, 1.0)
    w = float(x[2])
    if abs(w) < 1e-12:
        w = 1e-12
    u = x[0] / w
    v = x[1] / w
    return float(u), float(v), w
