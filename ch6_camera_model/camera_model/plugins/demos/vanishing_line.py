"""
Vanishing line demo: input plane normal (nx, ny, nz) in world frame;
output vanishing line l_inf = [lx, ly, lz] in homogeneous P2 and plot it on the image.

Formula: l = K^{-T} R n where n is the plane normal in world frame, R = R_world_to_cam.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFormLayout,
    QLabel,
    QDoubleSpinBox,
)

from ..registry import Demo


def _line_segment_in_rect(lx: float, ly: float, lz: float, w: float, h: float):
    """
    Intersect line lx*u + ly*v + lz = 0 with rectangle [0,w] x [0,h].
    Returns list of (u, v) points on the boundary (0–2 points for visible segment).
    """
    pts = []
    if abs(ly) >= 1e-10:
        for u in (0.0, w):
            v = -(lx * u + lz) / ly
            if 0 <= v <= h:
                pts.append((u, v))
    if abs(lx) >= 1e-10:
        for v in (0.0, h):
            u = -(ly * v + lz) / lx
            if 0 <= u <= w:
                pts.append((u, v))
    # Deduplicate and order for drawing (optional: sort by angle for a single segment)
    seen = set()
    unique = []
    for p in pts:
        key = (round(p[0], 6), round(p[1], 6))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if len(unique) >= 2:
        return unique[:2]
    return unique


class VanishingLineDemo(Demo):
    """Compute and display the vanishing line of a plane given its normal (world frame)."""

    def __init__(self):
        self._l_inf: Optional[np.ndarray] = None  # (3,) homogeneous line
        self._output_label: Optional[QLabel] = None
        self._spin_nx: Optional[QDoubleSpinBox] = None
        self._spin_ny: Optional[QDoubleSpinBox] = None
        self._spin_nz: Optional[QDoubleSpinBox] = None

    def id(self) -> str:
        return "vanishing_line"

    def label(self) -> str:
        return "Vanishing line"

    def on_activated(self, context: dict) -> None:
        self._l_inf = None
        self._update_output_display()

    def on_deactivated(self) -> None:
        self._l_inf = None
        self._update_output_display()

    def _update_output_display(self) -> None:
        if self._output_label is None:
            return
        if self._l_inf is None:
            self._output_label.setText("l_inf = [—] (click Update)")
        else:
            l = self._l_inf
            self._output_label.setText(f"l_inf = [{l[0]:.4g}, {l[1]:.4g}, {l[2]:.4g}]")

    def _on_update(self) -> None:
        if self._spin_nx is None or self._spin_ny is None or self._spin_nz is None:
            return
        # Get context from the pane's stored reference (set when pane is built)
        ctx = getattr(self, "_pane_context", None)
        if ctx is None:
            return
        state = ctx.get("state")
        if state is None:
            return
        K = state.get_K()
        R_cw, _ = state.get_R_and_t()
        nx = self._spin_nx.value()
        ny = self._spin_ny.value()
        nz = self._spin_nz.value()
        n_world = np.array([nx, ny, nz], dtype=np.float64)
        n_norm = np.linalg.norm(n_world)
        if n_norm < 1e-10:
            self._l_inf = None
            self._update_output_display()
            redraw = ctx.get("request_redraw")
            if callable(redraw):
                redraw()
            return
        n_world = n_world / n_norm
        n_cam = R_cw @ n_world
        K_inv_T = np.linalg.inv(K).T
        self._l_inf = K_inv_T @ n_cam
        # Normalize for display (optional)
        s = np.linalg.norm(self._l_inf)
        if s > 1e-10:
            self._l_inf = self._l_inf / s
        self._update_output_display()
        redraw = ctx.get("request_redraw")
        if callable(redraw):
            redraw()

    def get_pane_widget(self, context: dict) -> Optional[QWidget]:
        self._pane_context = context
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()
        self._spin_nx = QDoubleSpinBox()
        self._spin_nx.setRange(-10.0, 10.0)
        self._spin_nx.setDecimals(4)
        self._spin_nx.setSingleStep(0.1)
        self._spin_nx.setValue(0.0)
        form.addRow(QLabel("nx:"), self._spin_nx)
        self._spin_ny = QDoubleSpinBox()
        self._spin_ny.setRange(-10.0, 10.0)
        self._spin_ny.setDecimals(4)
        self._spin_ny.setSingleStep(0.1)
        self._spin_ny.setValue(1.0)
        form.addRow(QLabel("ny:"), self._spin_ny)
        self._spin_nz = QDoubleSpinBox()
        self._spin_nz.setRange(-10.0, 10.0)
        self._spin_nz.setDecimals(4)
        self._spin_nz.setSingleStep(0.1)
        self._spin_nz.setValue(0.0)
        form.addRow(QLabel("nz:"), self._spin_nz)
        layout.addLayout(form)
        btn = QPushButton("Update")
        btn.setToolTip("Compute vanishing line l_inf = K^{-T} R n and plot on image")
        btn.clicked.connect(self._on_update)
        layout.addWidget(btn)
        self._output_label = QLabel("l_inf = [—] (click Update)")
        self._output_label.setWordWrap(True)
        layout.addWidget(self._output_label)
        layout.addStretch()
        return widget

    def on_draw_image(self, ax_img, context: dict) -> None:
        if self._l_inf is None:
            return
        w = context.get("image_width_px", 1)
        h = context.get("image_height_px", 1)
        if w <= 0 or h <= 0:
            return
        lx, ly, lz = float(self._l_inf[0]), float(self._l_inf[1]), float(self._l_inf[2])
        pts = _line_segment_in_rect(lx, ly, lz, float(w), float(h))
        if len(pts) >= 2:
            u = [pts[0][0], pts[1][0]]
            v = [pts[0][1], pts[1][1]]
            ax_img.plot(u, v, "c-", linewidth=2, zorder=6, label="Vanishing line")
        elif len(pts) == 1:
            ax_img.scatter(pts[0][0], pts[0][1], c="c", s=60, zorder=6)
