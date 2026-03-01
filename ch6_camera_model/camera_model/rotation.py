"""
Camera/world rotation conventions and conversions (yaw, pitch, roll).

Single responsibility: euler angle ↔ rotation matrix for camera pose.
"""
from __future__ import annotations

import numpy as np


def R_wc_from_yaw_pitch_roll_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build R_wb (3x3) which represents the orientation of the body in world coordinates.

    Camera axes: X right, Y down, Z forward.
    Rotations: yaw about +Y, pitch about +X, roll about +Z.
    Composition: R_wc = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Rz_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0.0],
        [np.sin(roll), np.cos(roll), 0.0],
        [0.0, 0.0, 1.0],
    ])
    Rx_pitch = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ])
    Ry_yaw = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ])
    return Ry_yaw @ Rx_pitch @ Rz_roll


def R_wb_from_yaw_pitch_roll_world(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Build R_wb (3x3): orientation of body in world coordinates.

    World axes: X right, Y forward, Z up.
    Rotations: yaw about +Z, pitch about +X, roll about +Y.
    Composition: R_wb = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    """
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ])
    Ry = np.array([
        [np.cos(roll), 0.0, np.sin(roll)],
        [0.0, 1.0, 0.0],
        [-np.sin(roll), 0.0, np.cos(roll)],
    ])
    return Rz @ Rx @ Ry


# Base rotation: camera default (yaw=pitch=roll=0) has optical axis along world -Y.
R_CW_BASE_CAM = R_wb_from_yaw_pitch_roll_world(0.0, -90.0, 0.0).T


def pitch_yaw_roll_from_R(R_world_to_cam: np.ndarray) -> tuple[float, float, float]:
    """
    Recover (pitch, yaw, roll) in degrees from R_world_to_cam.
    Convention: same as R_wc_from_yaw_pitch_roll_camera; R_wc = Ry(yaw) @ Rx(pitch) @ Rz(roll).
    """
    R_bw = R_world_to_cam @ R_CW_BASE_CAM.T
    sp = -R_bw[2, 1]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)

    if abs(cp) > 1e-6:
        roll = np.arctan2(R_bw[0, 1], R_bw[1, 1])
        yaw = np.arctan2(R_bw[2, 0], R_bw[2, 2])
    else:
        roll = 0.0
        yaw = np.arctan2(-R_bw[1, 0], R_bw[0, 0])

    return (
        np.rad2deg(pitch),
        np.rad2deg(yaw),
        np.rad2deg(roll),
    )
