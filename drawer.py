import os

# Enable headless mode to allow running without a display
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from PyQt5.QtCore import Qt

from constants import EMPTY, FULL, UNKNOWN


class QtDrawer:
    """Simple PyQt5 drawer for the Nonogram solver."""

    def __init__(self, cell_size: int = 20):
        self.app = QApplication.instance() or QApplication([])
        self.cell = cell_size

        # dimensions will be initialised on first draw
        self.height = None
        self.width = None

        # progress window
        self.progress_label = QLabel()
        self.progress_label.setWindowTitle("Progress")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.hide()

        # solutions window
        self.solution_widget = QWidget()
        self.solution_widget.setWindowTitle("Solutions")
        self.solution_label = QLabel()
        self.solution_label.setAlignment(Qt.AlignCenter)
        self.prev_button = QPushButton("◀")
        self.next_button = QPushButton("▶")
        self.prev_button.clicked.connect(self.prev_solution)
        self.next_button.clicked.connect(self.next_solution)
        buttons = QHBoxLayout()
        buttons.addWidget(self.prev_button)
        buttons.addWidget(self.next_button)
        layout = QVBoxLayout()
        layout.addWidget(self.solution_label)
        layout.addLayout(buttons)
        self.solution_widget.setLayout(layout)
        self.solution_widget.hide()

        self.solutions = []  # list of numpy arrays representing solutions
        self.index = -1
        self._update_button_visibility()

    # ------------------------------------------------------------------
    # public API used by solver
    def update_progress(self, pic):
        """Update the progress window with the current picture."""
        self._ensure_size(pic)
        self.progress_label.setPixmap(self._pixmap_from_picture(pic))
        self.progress_label.show()
        self.app.processEvents()

    def add_solution(self, pic):
        """Store a newly found solution and update the solution window."""
        self._ensure_size(pic)
        self.solutions.append(pic.get_pixels().copy())
        self.index = len(self.solutions) - 1
        self.solution_widget.show()
        self._display_current_solution()
        self.app.processEvents()

    # ------------------------------------------------------------------
    # navigation callbacks
    def next_solution(self):
        if self.index < len(self.solutions) - 1:
            self.index += 1
            self._display_current_solution()

    def prev_solution(self):
        if self.index > 0:
            self.index -= 1
            self._display_current_solution()

    # ------------------------------------------------------------------
    # helpers
    def _ensure_size(self, pic):
        if self.height != pic.height or self.width != pic.width:
            self.height = pic.height
            self.width = pic.width
            w, h = self.width * self.cell, self.height * self.cell
            self.progress_label.resize(w, h)
            self.solution_label.resize(w, h)

    def _display_current_solution(self):
        if not self.solutions:
            return
        arr = self.solutions[self.index]
        self.solution_label.setPixmap(self._pixmap_from_array(arr))
        self._update_button_visibility()

    def _update_button_visibility(self):
        multiple = len(self.solutions) > 1
        self.prev_button.setVisible(multiple)
        self.next_button.setVisible(multiple)
        self.prev_button.setEnabled(multiple and self.index > 0)
        self.next_button.setEnabled(multiple and self.index < len(self.solutions) - 1)

    def _pixmap_from_picture(self, pic):
        return self._pixmap_from_array(pic.get_pixels())

    def _pixmap_from_array(self, arr):
        h, w = arr.shape
        img = QImage(w * self.cell, h * self.cell, QImage.Format_RGB32)
        painter = QPainter(img)
        colors = {
            EMPTY: QColor("white"),
            FULL: QColor("black"),
            UNKNOWN: QColor("gray"),
        }
        for i in range(h):
            for j in range(w):
                painter.fillRect(
                    j * self.cell,
                    i * self.cell,
                    self.cell,
                    self.cell,
                    colors.get(int(arr[i, j]), QColor("gray")),
                )
        painter.end()
        return QPixmap.fromImage(img)


__all__ = ["QtDrawer"]

