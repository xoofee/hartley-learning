"""
App-wide log sink: route log messages to a UI widget (or fallback to print).

Usage:
  from camera_model.logging_ui import log, set_log_sink

  set_log_sink(my_log_widget)  # once at startup
  log("hello")                 # anywhere in the app
"""
from __future__ import annotations

import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QPlainTextEdit

_sink: QPlainTextEdit | None = None
_use_timestamp: bool = True


def set_log_sink(widget: QPlainTextEdit | None, use_timestamp: bool = True) -> None:
    """Register the widget that receives log messages. Pass None to clear."""
    global _sink, _use_timestamp
    _sink = widget
    _use_timestamp = use_timestamp


def get_log_sink() -> QPlainTextEdit | None:
    """Return the current log sink widget, or None."""
    return _sink


def log(message: str, timestamp: bool | None = None) -> None:
    """
    Append a message to the app log output (or print if no sink is set).

    message: text to log (can include newlines).
    timestamp: if True/False, override default; if None, use sink default.
    """
    global _use_timestamp
    ts = (timestamp if timestamp is not None else _use_timestamp) and _sink is not None
    prefix = f"[{datetime.now().strftime('%H:%M:%S')}] " if ts else ""
    line = f"{prefix}{message}"
    if _sink is not None:
        _sink.appendPlainText(line)
        # keep cursor at end and ensure visible
        cursor = _sink.textCursor()
        cursor.movePosition(cursor.End)
        _sink.setTextCursor(cursor)
        _sink.ensureCursorVisible()
    else:
        print(line, file=sys.stderr, flush=True)


def log_exception(exc: BaseException, context: str = "") -> None:
    """Log an exception and its traceback. Optional context string prefix."""
    import traceback
    msg = f"{context}\n" if context else ""
    msg += "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log(msg.strip(), timestamp=True)
