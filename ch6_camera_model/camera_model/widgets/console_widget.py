"""
Interactive Python console: single widget, like a real REPL.
Input and output in one area; you type after the prompt; Enter runs.
"""
from __future__ import annotations

import sys
import code
from io import StringIO
from typing import Callable

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor


class ConsoleWidget(QWidget):
    """
    Single-widget REPL: one text area. History (prompts + output) and current input
    are in the same document; editing is only allowed after the last prompt.
    Enter runs the current input.
    """

    def __init__(
        self,
        namespace_getter: Callable[[], dict],
        title: str = "Console",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._namespace_getter = namespace_getter
        self._title = title
        self._buffer = ""
        self._input_start_pos = 0
        self._persistent_ns: dict | None = None
        self._console: code.InteractiveConsole | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if title:
            label = QLabel(title)
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label)

        self._te = QPlainTextEdit(self)
        self._te.setFont(QFont("Consolas", 9))
        self._te.setPlainText(">>> ")
        self._input_start_pos = 4
        self._te.installEventFilter(self)
        layout.addWidget(self._te)

    def _get_input_text(self) -> str:
        doc = self._te.document()
        return doc.toPlainText()[self._input_start_pos:]

    def _replace_input_region(self, new_text: str, prompt_suffix: str = ">>> ") -> None:
        """Replace from _input_start_pos to end with new_text + prompt; update _input_start_pos."""
        c = self._te.textCursor()
        c.setPosition(self._input_start_pos)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        c.insertText(new_text + prompt_suffix)
        self._input_start_pos = c.position()  # editable part starts after the prompt
        self._te.setTextCursor(c)
        self._te.ensureCursorVisible()

    def eventFilter(self, obj, event) -> bool:
        if obj is self._te and event.type() == event.KeyPress:
            key = event.key()
            cursor = self._te.textCursor()
            pos = cursor.position()
            if key == Qt.Key_Return or key == Qt.Key_Enter:
                text = self._get_input_text()
                if not text.strip() and not self._buffer:
                    # Empty line: insert newline and "... " for continuation
                    self._replace_input_region("\n", "... ")
                    return True
                self._buffer += text if text.endswith("\n") else text + "\n"
                # Persistent namespace: refresh state/P/K/... from getter, keep imports and user vars
                fresh = self._namespace_getter()
                if self._persistent_ns is None:
                    self._persistent_ns = dict(fresh)
                    self._console = code.InteractiveConsole(locals=self._persistent_ns)
                else:
                    self._persistent_ns.update(fresh)
                old_stdout, old_stderr = sys.stdout, sys.stderr
                stream = StringIO()
                sys.stdout = stream
                sys.stderr = stream
                try:
                    more = self._console.push(self._buffer)
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
                out = stream.getvalue()
                prompt = "... " if more else ">>> "
                # Show input (with ... for continuation lines only); leading >>> is already in doc
                input_display = self._buffer.strip().replace("\n", "\n... ") + "\n"
                new_content = input_display + ((out.rstrip() + "\n") if out else "")
                if not more:
                    self._buffer = ""
                self._replace_input_region(new_content, prompt)
                return True
            # Block editing in history; allow selection (so history is copyable)
            if key == Qt.Key_Backspace:
                if pos <= self._input_start_pos or cursor.anchor() < self._input_start_pos:
                    return True
            elif key == Qt.Key_Delete:
                if pos < self._input_start_pos or cursor.anchor() < self._input_start_pos:
                    return True
            elif event.text():
                # Character input: if selection, append at end of input; if in history, move to input
                c = self._te.textCursor()
                if c.hasSelection():
                    c.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)
                    self._te.setTextCursor(c)
                elif pos < self._input_start_pos:
                    c.setPosition(self._input_start_pos)
                    self._te.setTextCursor(c)
            elif (event.modifiers() & Qt.ControlModifier) and key in (Qt.Key_V, Qt.Key_X):
                # Paste/cut: if cursor in history, move to input area
                if pos < self._input_start_pos:
                    c = self._te.textCursor()
                    c.setPosition(self._input_start_pos)
                    self._te.setTextCursor(c)
        return super().eventFilter(obj, event)

    def append_plain_text(self, text: str) -> None:
        c = self._te.textCursor()
        c.movePosition(QTextCursor.End)
        c.insertText(text.rstrip() + "\n")
        self._input_start_pos = c.position()
        self._te.setTextCursor(c)

    def clear_output(self) -> None:
        self._te.clear()
        self._te.setPlainText(">>> ")
        self._input_start_pos = 4
        self._buffer = ""
        self._persistent_ns = None
        self._console = None
