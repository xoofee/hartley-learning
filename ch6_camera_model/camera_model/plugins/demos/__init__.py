"""
Built-in demos: exclusive modes (None, P row planes, Backproject, Angulometer).

Each demo is independent; when its button is off, on_deactivated() releases all state.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QButtonGroup, QPushButton

from ..registry import register_demo, get_demos
from .none_demo import NoneDemo
from .p_row_planes import PRowPlanesDemo
from .backproject import BackprojectDemo
from .angulometer import AngulometerDemo


def build_demos_button_group(parent) -> tuple[QGroupBox, QButtonGroup, dict[str, QPushButton]]:
    """Create a Demos group with exclusive buttons. Returns (group_widget, button_group, id_to_button)."""
    group = QGroupBox("Demos")
    layout = QVBoxLayout()
    button_group = QButtonGroup(parent)
    button_group.setExclusive(True)
    id_to_button: dict[str, QPushButton] = {}
    for demo in get_demos():
        btn = QPushButton(demo.label())
        btn.setCheckable(True)
        if demo.id() == "none":
            btn.setChecked(True)
        btn.setProperty("demo_id", demo.id())
        button_group.addButton(btn)
        layout.addWidget(btn)
        id_to_button[demo.id()] = btn
    group.setLayout(layout)
    return group, button_group, id_to_button


def register_builtin_demos() -> None:
    """Register None, P row planes, Backproject, Angulometer demos."""
    register_demo(NoneDemo())
    register_demo(PRowPlanesDemo())
    register_demo(BackprojectDemo())
    register_demo(AngulometerDemo())


__all__ = [
    "NoneDemo",
    "PRowPlanesDemo",
    "BackprojectDemo",
    "AngulometerDemo",
    "build_demos_button_group",
    "register_builtin_demos",
]
