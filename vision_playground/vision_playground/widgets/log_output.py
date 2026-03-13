"""
Log output widget: read-only text area for app log messages with level-based display filter.
Stores (level, line) and filters what is shown. Implements append_log(level, line) for logging_ui.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Level constants must match logging_ui (DEBUG=0, INFO=1, WARNING=2, ERROR=3)
DISPLAY_FILTER_DEBUG = 0   # show all
DISPLAY_FILTER_INFO = 1
DISPLAY_FILTER_WARNING = 2
DISPLAY_FILTER_ERROR = 3   # show only ERROR


class LogOutputWidget(QWidget):
    """Read-only log output with optional clear button and display filter (by level)."""

    def __init__(
        self,
        title: str = "Log",
        show_clear_button: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._lines: list[tuple[int, str]] = []  # (level, text)
        self._display_filter_level = DISPLAY_FILTER_DEBUG
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        if title:
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold;")
            header.addWidget(label)
        header.addStretch()
        self._filter_combo = QComboBox()
        self._filter_combo.setToolTip("Filter which log levels are shown in the log panel.")
        self._filter_combo.addItem("All (Debug+)", DISPLAY_FILTER_DEBUG)
        self._filter_combo.addItem("Info+", DISPLAY_FILTER_INFO)
        self._filter_combo.addItem("Warning+", DISPLAY_FILTER_WARNING)
        self._filter_combo.addItem("Error only", DISPLAY_FILTER_ERROR)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        header.addWidget(QLabel("Show:"))
        header.addWidget(self._filter_combo)
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

    def _on_filter_changed(self, index: int) -> None:
        self._display_filter_level = self._filter_combo.currentData()
        if self._display_filter_level is None:
            self._display_filter_level = DISPLAY_FILTER_DEBUG
        self._refresh_display()
        if hasattr(self, "_on_display_filter_changed_cb") and callable(self._on_display_filter_changed_cb):
            self._on_display_filter_changed_cb()

    def _refresh_display(self) -> None:
        visible = [
            line for level, line in self._lines
            if level >= self._display_filter_level
        ]
        self._text.setPlainText("\n".join(visible))
        cursor = self._text.textCursor()
        cursor.movePosition(cursor.End)
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

    def append_log(self, level: int, line: str) -> None:
        """Append a log line with level (used by logging_ui when level >= minimum)."""
        self._lines.append((level, line))
        if level >= self._display_filter_level:
            self._text.appendPlainText(line)
            cursor = self._text.textCursor()
            cursor.movePosition(cursor.End)
            self._text.setTextCursor(cursor)
            self._text.ensureCursorVisible()

    def appendPlainText(self, text: str) -> None:
        """Backward compat: append as INFO level."""
        self.append_log(DISPLAY_FILTER_INFO, text)

    def set_display_filter_level(self, level: int) -> None:
        """Set display filter (DEBUG=0, INFO=1, WARNING=2, ERROR=3)."""
        self._display_filter_level = level
        idx = self._filter_combo.findData(level)
        if idx >= 0:
            self._filter_combo.blockSignals(True)
            self._filter_combo.setCurrentIndex(idx)
            self._filter_combo.blockSignals(False)
        self._refresh_display()

    def get_display_filter_level(self) -> int:
        """Return current display filter level."""
        d = self._filter_combo.currentData()
        return int(d) if d is not None else DISPLAY_FILTER_DEBUG

    def set_display_filter_changed_callback(self, callback) -> None:
        """Set a callback when the user changes the display filter (for persistence)."""
        self._on_display_filter_changed_cb = callback

    def clear(self) -> None:
        self._lines.clear()
        self._text.clear()

    def toPlainText(self) -> str:
        return self._text.toPlainText()

    def textCursor(self):
        return self._text.textCursor()

    def setTextCursor(self, cursor) -> None:
        self._text.setTextCursor(cursor)

    def ensureCursorVisible(self) -> None:
        self._text.ensureCursorVisible()
