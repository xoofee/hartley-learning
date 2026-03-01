"""Camera intrinsics and distortion parameter widgets."""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QDoubleSpinBox,
)
from PyQt5.QtGui import QFont

from ..state import CameraState


class CameraParamsWidget(QWidget):
    """Editable intrinsics: focal_length_mm, sensor_*, pixel_*."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("Camera parameters")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout()
        grid = QGridLayout()
        row = 0
        grid.addWidget(QLabel("focal_length"), row, 0)
        self.spin_f = QDoubleSpinBox()
        self.spin_f.setRange(1.0, 5000.0)
        self.spin_f.setSuffix(" mm")
        self.spin_f.setValue(state.focal_length_mm)
        self.spin_f.setMaximumWidth(90)
        grid.addWidget(self.spin_f, row, 1)
        row += 1
        grid.addWidget(QLabel("sensor_width"), row, 0)
        self.spin_wphys = QDoubleSpinBox()
        self.spin_wphys.setRange(0.1, 200.0)
        self.spin_wphys.setDecimals(2)
        self.spin_wphys.setSuffix(" mm")
        self.spin_wphys.setValue(state.sensor_width_mm)
        self.spin_wphys.setMaximumWidth(90)
        grid.addWidget(self.spin_wphys, row, 1)
        row += 1
        grid.addWidget(QLabel("sensor_height"), row, 0)
        self.spin_hphys = QDoubleSpinBox()
        self.spin_hphys.setRange(0.1, 200.0)
        self.spin_hphys.setDecimals(2)
        self.spin_hphys.setSuffix(" mm")
        self.spin_hphys.setValue(state.sensor_height_mm)
        self.spin_hphys.setMaximumWidth(90)
        grid.addWidget(self.spin_hphys, row, 1)
        row += 1
        grid.addWidget(QLabel("pixel_size_x"), row, 0)
        self.spin_wpix = QDoubleSpinBox()
        self.spin_wpix.setRange(0.0001, 1.0)
        self.spin_wpix.setDecimals(4)
        self.spin_wpix.setSuffix(" mm")
        self.spin_wpix.setValue(state.pixel_size_x_mm)
        self.spin_wpix.setMaximumWidth(90)
        grid.addWidget(self.spin_wpix, row, 1)
        row += 1
        grid.addWidget(QLabel("pixel_size_y"), row, 0)
        self.spin_hpix = QDoubleSpinBox()
        self.spin_hpix.setRange(0.0001, 1.0)
        self.spin_hpix.setDecimals(4)
        self.spin_hpix.setSuffix(" mm")
        self.spin_hpix.setValue(state.pixel_size_y_mm)
        self.spin_hpix.setMaximumWidth(90)
        grid.addWidget(self.spin_hpix, row, 1)
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.focal_length_mm = self.spin_f.value()
        self.state.sensor_width_mm = self.spin_wphys.value()
        self.state.sensor_height_mm = self.spin_hphys.value()
        self.state.pixel_size_x_mm = self.spin_wpix.value()
        self.state.pixel_size_y_mm = self.spin_hpix.value()

    def sync_from_state(self) -> None:
        for spin in (self.spin_f, self.spin_wphys, self.spin_hphys, self.spin_wpix, self.spin_hpix):
            spin.blockSignals(True)
        self.spin_f.setValue(self.state.focal_length_mm)
        self.spin_wphys.setValue(self.state.sensor_width_mm)
        self.spin_hphys.setValue(self.state.sensor_height_mm)
        self.spin_wpix.setValue(self.state.pixel_size_x_mm)
        self.spin_hpix.setValue(self.state.pixel_size_y_mm)
        for spin in (self.spin_f, self.spin_wphys, self.spin_hphys, self.spin_wpix, self.spin_hpix):
            spin.blockSignals(False)


class DistortionParamsWidget(QWidget):
    """Lens distortion (OpenCV-style): k1, k2, k3 radial; p1, p2 tangential."""

    def __init__(self, state: CameraState):
        super().__init__()
        self.state = state
        group = QGroupBox("Lens distortion")
        group.setFont(QFont("Arial", 10, QFont.Bold))
        grid = QGridLayout()
        row = 0
        for label, attr, decimals in [
            ("k1:", "dist_k1", 4),
            ("k2:", "dist_k2", 6),
            ("k3:", "dist_k3", 6),
            ("p1:", "dist_p1", 6),
            ("p2:", "dist_p2", 6),
        ]:
            grid.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-1.0, 1.0)
            spin.setDecimals(decimals)
            spin.setSingleStep(0.01 if "k" in label else 0.001)
            spin.setValue(getattr(state, attr))
            spin.setMaximumWidth(90)
            grid.addWidget(spin, row, 1)
            setattr(self, "spin_" + attr, spin)
            row += 1
        layout = QVBoxLayout()
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def apply_to_state(self) -> None:
        self.state.dist_k1 = self.spin_dist_k1.value()
        self.state.dist_k2 = self.spin_dist_k2.value()
        self.state.dist_k3 = self.spin_dist_k3.value()
        self.state.dist_p1 = self.spin_dist_p1.value()
        self.state.dist_p2 = self.spin_dist_p2.value()

    def sync_from_state(self) -> None:
        for spin in (
            self.spin_dist_k1,
            self.spin_dist_k2,
            self.spin_dist_k3,
            self.spin_dist_p1,
            self.spin_dist_p2,
        ):
            spin.blockSignals(True)
        self.spin_dist_k1.setValue(self.state.dist_k1)
        self.spin_dist_k2.setValue(self.state.dist_k2)
        self.spin_dist_k3.setValue(self.state.dist_k3)
        self.spin_dist_p1.setValue(self.state.dist_p1)
        self.spin_dist_p2.setValue(self.state.dist_p2)
        for spin in (
            self.spin_dist_k1,
            self.spin_dist_k2,
            self.spin_dist_k3,
            self.spin_dist_p1,
            self.spin_dist_p2,
        ):
            spin.blockSignals(False)
