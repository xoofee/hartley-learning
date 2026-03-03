"""
Log output widget: read-only text area for app log messages.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class LogOutputWidget(QWidget):
    """Read-only log output with optional clear button."""

    def __init__(
        self,
        title: str = "Log",
        show_clear_button: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if title or show_clear_button:
            header = QHBoxLayout()
            if title:
                label = QLabel(title)
                label.setStyleSheet("font-weight: bold;")
                header.addWidget(label)
            header.addStretch()
            if show_clear_button:
                clear_btn = QPushButton("Clear")
                clear_btn.setMaximumWidth(60)
                clear_btn.clicked.connect(self.clear)
                header.addWidget(clear_btn)
            layout.addLayout(header)
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)
        self._text.setPlaceholderText("Log output appears here…")
        self._text.setFont(QFont("Consolas", 9))
        layout.addWidget(self._text)

    def appendPlainText(self, text: str) -> None:
        self._text.appendPlainText(text)

    def clear(self) -> None:
        self._text.clear()

    def toPlainText(self) -> str:
        return self._text.toPlainText()

    def textCursor(self):
        return self._text.textCursor()

    def setTextCursor(self, cursor) -> None:
        self._text.setTextCursor(cursor)

    def ensureCursorVisible(self) -> None:
        self._text.ensureCursorVisible()
