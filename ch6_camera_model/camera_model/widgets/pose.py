"""Pose widgets: pitch/yaw/roll and camera center C."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
)
from PyQt5.QtGui import QFont

from ..state import CameraState


class RotationParamsWidget(QWidget):
    """Pitch, yaw, roll (°)."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("Pitch / Yaw / Roll")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        grid = QGridLayout()
        grid.addWidget(QLabel("pitch:"), 0, 0)
        self.spin_pitch = QDoubleSpinBox()
        self.spin_pitch.setRange(-90.0, 90.0)
        self.spin_pitch.setDecimals(1)
        self.spin_pitch.setSuffix(" °")
        self.spin_pitch.setValue(state.pitch_deg)
        self.spin_pitch.setMaximumWidth(72)
        grid.addWidget(self.spin_pitch, 0, 1)
        grid.addWidget(QLabel("yaw:"), 1, 0)
        self.spin_yaw = QDoubleSpinBox()
        self.spin_yaw.setRange(-360.0, 360.0)
        self.spin_yaw.setDecimals(1)
        self.spin_yaw.setSuffix(" °")
        self.spin_yaw.setValue(state.yaw_deg)
        self.spin_yaw.setMaximumWidth(72)
        grid.addWidget(self.spin_yaw, 1, 1)
        grid.addWidget(QLabel("roll:"), 2, 0)
        self.spin_roll = QDoubleSpinBox()
        self.spin_roll.setRange(-180.0, 180.0)
        self.spin_roll.setDecimals(1)
        self.spin_roll.setSuffix(" °")
        self.spin_roll.setValue(state.roll_deg)
        self.spin_roll.setMaximumWidth(72)
        grid.addWidget(self.spin_roll, 2, 1)
        group.setLayout(grid)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.pitch_deg = self.spin_pitch.value()
        self.state.yaw_deg = self.spin_yaw.value()
        self.state.roll_deg = self.spin_roll.value()

    def sync_from_state(self) -> None:
        for spin in (self.spin_pitch, self.spin_yaw, self.spin_roll):
            spin.blockSignals(True)
        self.spin_pitch.setValue(self.state.pitch_deg)
        self.spin_yaw.setValue(self.state.yaw_deg)
        self.spin_roll.setValue(self.state.roll_deg)
        for spin in (self.spin_pitch, self.spin_yaw, self.spin_roll):
            spin.blockSignals(False)


class CameraCenterWidget(QWidget):
    """Camera center C in world: C_x, C_y, C_z."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("C (world)")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        grid = QGridLayout()
        grid.addWidget(QLabel("C_x:"), 0, 0)
        self.spin_Cx = QDoubleSpinBox()
        self.spin_Cx.setRange(-100.0, 100.0)
        self.spin_Cx.setDecimals(3)
        self.spin_Cx.setSingleStep(0.1)
        self.spin_Cx.setValue(state.C_x)
        self.spin_Cx.setMaximumWidth(72)
        grid.addWidget(self.spin_Cx, 0, 1)
        grid.addWidget(QLabel("C_y:"), 1, 0)
        self.spin_Cy = QDoubleSpinBox()
        self.spin_Cy.setRange(-100.0, 100.0)
        self.spin_Cy.setDecimals(3)
        self.spin_Cy.setSingleStep(0.1)
        self.spin_Cy.setValue(state.C_y)
        self.spin_Cy.setMaximumWidth(72)
        grid.addWidget(self.spin_Cy, 1, 1)
        grid.addWidget(QLabel("C_z:"), 2, 0)
        self.spin_Cz = QDoubleSpinBox()
        self.spin_Cz.setRange(-100.0, 100.0)
        self.spin_Cz.setDecimals(3)
        self.spin_Cz.setSingleStep(0.1)
        self.spin_Cz.setValue(state.C_z)
        self.spin_Cz.setMaximumWidth(72)
        grid.addWidget(self.spin_Cz, 2, 1)
        group.setLayout(grid)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.C_x = self.spin_Cx.value()
        self.state.C_y = self.spin_Cy.value()
        self.state.C_z = self.spin_Cz.value()

    def sync_from_state(self) -> None:
        for spin in (self.spin_Cx, self.spin_Cy, self.spin_Cz):
            spin.blockSignals(True)
        self.spin_Cx.setValue(self.state.C_x)
        self.spin_Cy.setValue(self.state.C_y)
        self.spin_Cz.setValue(self.state.C_z)
        for spin in (self.spin_Cx, self.spin_Cy, self.spin_Cz):
            spin.blockSignals(False)
