"""
App-wide log sink: route log messages to a UI widget (or fallback to print).
Supports levels: DEBUG, INFO, WARNING, ERROR. Minimum level controls what is passed to the sink.

Usage:
  from vision_playground.logging_ui import log, log_debug, set_log_sink, set_minimum_level

  set_log_sink(my_log_widget)
  set_minimum_level(DEBUG)  # or INFO (default), WARNING, ERROR
  log("hello")              # INFO by default
  log_debug("detail")       # only appears if level is DEBUG
"""
from __future__ import annotations

import inspect
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QPlainTextEdit

# Level order: lower value = more verbose. Only messages with level >= minimum are sent to sink.
DEBUG = 0
INFO = 1
WARNING = 2
ERROR = 3

LEVEL_NAMES = {DEBUG: "DEBUG", INFO: "INFO", WARNING: "WARNING", ERROR: "ERROR"}

_sink: "QPlainTextEdit | None" = None
_use_timestamp: bool = True
_minimum_level: int = INFO


def set_log_sink(widget: "QPlainTextEdit | None", use_timestamp: bool = True) -> None:
    """Register the widget that receives log messages. Pass None to clear."""
    global _sink, _use_timestamp
    _sink = widget
    _use_timestamp = use_timestamp


def get_log_sink() -> "QPlainTextEdit | None":
    """Return the current log sink widget, or None."""
    return _sink


def set_minimum_level(level: int) -> None:
    """Set minimum log level (DEBUG, INFO, WARNING, ERROR). Only messages at or above this level are sent to the sink."""
    global _minimum_level
    _minimum_level = level


def get_minimum_level() -> int:
    """Return the current minimum log level."""
    return _minimum_level


def _caller_location() -> str:
    """Return 'source_file_name:line_number' for the first frame outside this module."""
    this_file = os.path.normcase(__file__)
    for frame_info in inspect.stack():
        filename = os.path.normcase(frame_info.filename)
        if filename != this_file:
            return f"{os.path.basename(frame_info.filename)}:{frame_info.lineno}"
    return "?:?"


def _format_line(level: int, message: str, timestamp: bool, location: str) -> str:
    ts = f"[{datetime.now().strftime('%H:%M:%S')}] " if timestamp else ""
    name = LEVEL_NAMES.get(level, "LOG")
    return f"{ts}[{name}] {location} {message}"


def log(message: str, level: int = INFO, timestamp: bool | None = None) -> None:
    """
    Append a message to the app log output (or print if no sink is set).
    Only forwarded to sink if level >= current minimum level.

    message: text to log (can include newlines).
    level: DEBUG, INFO, WARNING, or ERROR (default INFO).
    timestamp: if True/False, override default; if None, use sink default.
    """
    global _use_timestamp
    if level < _minimum_level:
        return
    ts = (timestamp if timestamp is not None else _use_timestamp) and _sink is not None
    location = _caller_location()
    line = _format_line(level, message, ts, location)
    if _sink is not None:
        if hasattr(_sink, "append_log"):
            _sink.append_log(level, line)
        else:
            _sink.appendPlainText(line)
            cursor = _sink.textCursor()
            cursor.movePosition(cursor.End)
            _sink.setTextCursor(cursor)
            _sink.ensureCursorVisible()
    else:
        print(line, file=sys.stderr, flush=True)


def log_debug(message: str, **kwargs) -> None:
    """Log at DEBUG level."""
    log(message, level=DEBUG, **kwargs)


def log_info(message: str, **kwargs) -> None:
    """Log at INFO level."""
    log(message, level=INFO, **kwargs)


def log_warning(message: str, **kwargs) -> None:
    """Log at WARNING level."""
    log(message, level=WARNING, **kwargs)


def log_error(message: str, **kwargs) -> None:
    """Log at ERROR level."""
    log(message, level=ERROR, **kwargs)


def log_exception(exc: BaseException, context: str = "") -> None:
    """Log an exception and its traceback at ERROR level. Optional context string prefix."""
    import traceback
    msg = f"{context}\n" if context else ""
    msg += "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log(msg.strip(), level=ERROR, timestamp=True)
