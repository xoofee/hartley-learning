"""
Camera state: intrinsics, pose (pitch/yaw/roll, C), distortion.

Single responsibility: hold editable camera parameters and derive K, R, t, P.
"""
from __future__ import annotations

import numpy as np

from .rotation import R_wc_from_yaw_pitch_roll_camera, R_CW_BASE_CAM, pitch_yaw_roll_from_R
from . import pinhole


class CameraState:
    """Mutable camera state: intrinsics, pose (pitch/yaw/roll, C), lens distortion."""

    def __init__(
        self,
        focal_length_mm: float = 50.0,
        sensor_width_mm: float = 36.0,
        sensor_height_mm: float = 24.0,
        pixel_size_x_mm: float = 0.01,
        pixel_size_y_mm: float = 0.01,
    ):
        self.focal_length_mm = focal_length_mm
        self.sensor_width_mm = sensor_width_mm
        self.sensor_height_mm = sensor_height_mm
        self.pixel_size_x_mm = pixel_size_x_mm
        self.pixel_size_y_mm = pixel_size_y_mm
        self.cx_px_override: float | None = None
        self.cy_px_override: float | None = None
        self.pitch_deg = -30.0
        self.yaw_deg = -90.0
        self.roll_deg = 0.0
        self.C_x = 4.0
        self.C_y = 0.0
        self.C_z = 2.0
        self.dist_k1 = 0.0
        self.dist_k2 = 0.0
        self.dist_k3 = 0.0
        self.dist_p1 = 0.0
        self.dist_p2 = 0.0

    def get_distortion(self) -> tuple[float, float, float, float, float]:
        """Return (k1, k2, k3, p1, p2) for lens distortion."""
        return (self.dist_k1, self.dist_k2, self.dist_k3, self.dist_p1, self.dist_p2)

    def get_R_cw(self) -> np.ndarray:
        return R_wc_from_yaw_pitch_roll_camera(
            self.yaw_deg, self.pitch_deg, self.roll_deg
        ).T @ R_CW_BASE_CAM

    def get_camera_center_world(self) -> np.ndarray:
        return np.array([self.C_x, self.C_y, self.C_z])

    def get_K(self) -> np.ndarray:
        return pinhole.build_intrinsic(
            self.focal_length_mm,
            self.sensor_width_mm,
            self.sensor_height_mm,
            self.pixel_size_x_mm,
            self.pixel_size_y_mm,
            cx_px=self.cx_px_override,
            cy_px=self.cy_px_override,
        )

    def get_P(self) -> np.ndarray:
        camera_center_world = self.get_camera_center_world()
        R_cam = self.get_R_cw()
        K = self.get_K()
        return pinhole.build_projection_matrix(K, R_cam, camera_center_world)

    def get_R_and_t(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (R_world_to_cam, t_world_to_cam). t = -R @ C."""
        R_cw = self.get_R_cw()
        C = np.array([[self.C_x], [self.C_y], [self.C_z]])
        t_world_to_cam = -R_cw @ C
        return R_cw, t_world_to_cam

    def set_from_K(self, K: np.ndarray) -> None:
        """Update focal_length_mm and principal point (px) from intrinsic matrix K."""
        self.focal_length_mm = float(
            0.5 * (K[0, 0] * self.pixel_size_x_mm + K[1, 1] * self.pixel_size_y_mm)
        )
        self.cx_px_override = float(K[0, 2])
        self.cy_px_override = float(K[1, 2])

    def set_from_C(self, C_world: np.ndarray) -> None:
        """Set camera center in world from 3-vector."""
        C = np.asarray(C_world).ravel()[:3]
        self.C_x, self.C_y, self.C_z = float(C[0]), float(C[1]), float(C[2])

    def set_from_P(self, P: np.ndarray) -> None:
        """Update state from full P: decompose to K, R, t; set pitch/yaw/roll and C (from t)."""
        K_intrinsic, R_world_to_cam, t_world_to_cam = pinhole.decompose_P(P)
        self.set_from_K(K_intrinsic)
        self.pitch_deg, self.yaw_deg, self.roll_deg = pitch_yaw_roll_from_R(R_world_to_cam)
        C = (-R_world_to_cam.T @ t_world_to_cam).ravel()[:3]
        self.C_x, self.C_y, self.C_z = float(C[0]), float(C[1]), float(C[2])
