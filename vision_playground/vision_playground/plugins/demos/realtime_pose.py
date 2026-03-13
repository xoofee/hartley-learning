"""
Realtime camera pose demo: use calibrated K and live chessboard to estimate pose.
When active: hide calibration pyramids in 3D; show chessboard + single live camera.
"""
from __future__ import annotations

import cv2
import numpy as np

from ...state import AppState
from ...calibration import find_chessboard_corners
from ..registry import Demo


class RealtimePoseDemo(Demo):
    def id(self) -> str:
        return "realtime_pose"

    def label(self) -> str:
        return "Realtime pose"

    def hide_calibration_pyramids(self) -> bool:
        return True

    def on_activated(self, context: dict) -> None:
        state: AppState = context.get("state")
        if state is not None:
            state.realtime_pose = None

    def on_deactivated(self) -> None:
        pass

    def on_frame(self, frame_bgr: np.ndarray, context: dict) -> None:
        state: AppState = context.get("state")
        display = frame_bgr.copy()
        state.realtime_display_frame = display
        if state is None or state.calibration is None:
            return
        cal = state.calibration
        cb = state.chessboard
        cols, rows = cb.cols, cb.rows
        square_size = cb.square_size
        corners, found = find_chessboard_corners(frame_bgr, cols, rows)
        if found:
            cv2.drawChessboardCorners(display, (cols, rows), corners, True)
        if not found:
            state.realtime_pose = None
            return
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= square_size
        dist = cal.dist if cal.dist.size else None
        success, rvec, tvec = cv2.solvePnP(objp, corners, cal.K, dist)
        if not success:
            state.realtime_pose = None
            return
        R_cam, _ = cv2.Rodrigues(rvec)
        C = (-R_cam.T @ tvec).ravel()
        R_cam_to_world = R_cam.T
        t_cam_to_world = C
        state.realtime_pose = (R_cam_to_world, t_cam_to_world)
