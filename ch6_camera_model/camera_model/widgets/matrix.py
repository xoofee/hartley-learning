"""Read-only and editable matrix grid widgets."""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QGroupBox, QLineEdit
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

READONLY_BG = "background-color: #e0e0e0;"


def _fmt(x: float) -> str:
    return f"{x:.4f}"


class MatrixDisplayWidget(QWidget):
    """Read-only grid with gray background."""

    def __init__(self, title: str, nrows: int, ncols: int):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.edits: list[list[QLineEdit]] = []
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout()
        grid = QGridLayout()
        for i in range(nrows):
            row = []
            for j in range(ncols):
                edit = QLineEdit()
                edit.setReadOnly(True)
                edit.setStyleSheet(READONLY_BG)
                edit.setMaximumWidth(72)
                edit.setAlignment(Qt.AlignRight)
                grid.addWidget(edit, i, j)
                row.append(edit)
            self.edits.append(row)
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def set_matrix(self, M: np.ndarray) -> None:
        for i in range(min(self.nrows, M.shape[0])):
            for j in range(min(self.ncols, M.shape[1])):
                self.edits[i][j].setText(_fmt(float(M[i, j])))


class MatrixEditWidget(QWidget):
    """Editable grid; emits matrix when changed."""

    matrix_changed = pyqtSignal(object)

    def __init__(self, title: str, nrows: int, ncols: int):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.edits: list[list[QLineEdit]] = []
        self.updating = False
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Bold))
        layout = QVBoxLayout()
        grid = QGridLayout()
        for i in range(nrows):
            row = []
            for j in range(ncols):
                edit = QLineEdit()
                edit.setMaximumWidth(72)
                edit.setAlignment(Qt.AlignRight)
                edit.editingFinished.connect(self._on_edit)
                grid.addWidget(edit, i, j)
                row.append(edit)
            self.edits.append(row)
        layout.addLayout(grid)
        group.setLayout(layout)
        main = QVBoxLayout()
        main.addWidget(group)
        self.setLayout(main)

    def _on_edit(self) -> None:
        if self.updating:
            return
        try:
            M = self.get_matrix()
            if M is not None:
                self.matrix_changed.emit(M)
        except (ValueError, TypeError):
            pass

    def get_matrix(self) -> np.ndarray | None:
        try:
            M = np.zeros((self.nrows, self.ncols))
            for i in range(self.nrows):
                for j in range(self.ncols):
                    M[i, j] = float(self.edits[i][j].text())
            return M
        except (ValueError, TypeError):
            return None

    def set_matrix(self, M: np.ndarray) -> None:
        self.updating = True
        for i in range(min(self.nrows, M.shape[0])):
            for j in range(min(self.ncols, M.shape[1])):
                self.edits[i][j].setText(_fmt(float(M[i, j])))
        self.updating = False
