"""
Built-in demos: None (default), Realtime pose.
"""
from __future__ import annotations

from PyQt5.QtWidgets import QGroupBox, QButtonGroup, QPushButton

from ..registry import register_demo, get_demos
from .none_demo import NoneDemo
from .realtime_pose import RealtimePoseDemo
from .rotate_image import RotateImageDemo


def build_demos_button_group(parent) -> tuple[QGroupBox, QButtonGroup, dict[str, QPushButton]]:
    """Create a Demos group with exclusive buttons. Returns (group_widget, button_group, id_to_button)."""
    group = QGroupBox("Demos")
    from PyQt5.QtWidgets import QVBoxLayout
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
    """Register None and Realtime pose demos."""
    register_demo(NoneDemo())
    register_demo(RealtimePoseDemo())
    register_demo(RotateImageDemo())


__all__ = [
    "NoneDemo",
    "RealtimePoseDemo",
    "RotateImageDemo",
    "build_demos_button_group",
    "register_builtin_demos",
]
