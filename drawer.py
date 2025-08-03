import os
from multiprocessing import Process, Queue
from queue import Empty, Full

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from constants import EMPTY, FULL, UNKNOWN

# Enable headless mode to allow running without a display
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class _QtDrawer:
    """Simple PyQt5 drawer that performs the actual rendering."""

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
        self.prev_button = QPushButton("\u25C0")
        self.next_button = QPushButton("\u25B6")
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
    # public API used by worker process
    def update_progress(self, arr):
        """Update the progress window with the current picture array."""
        self._ensure_size(arr)
        self.progress_label.setPixmap(self._pixmap_from_array(arr))
        self.progress_label.show()

    def add_solution(self, arr):
        """Store a newly found solution and update the solution window."""
        self._ensure_size(arr)
        self.solutions.append(arr.copy())
        self.index = len(self.solutions) - 1
        self.solution_widget.show()
        self._display_current_solution()

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
    def _ensure_size(self, arr):
        h, w = arr.shape
        if self.height != h or self.width != w:
            self.height = h
            self.width = w
            w_px, h_px = self.width * self.cell, self.height * self.cell
            self.progress_label.resize(w_px, h_px)
            self.solution_label.resize(w_px, h_px)

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
        self.next_button.setEnabled(
            multiple and self.index < len(
                self.solutions) - 1)

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


def _drawer_process(queue: Queue, cell_size: int) -> None:
    drawer = _QtDrawer(cell_size)
    while True:
        try:
            msg, data = queue.get(timeout=0.05)
        except Empty:
            drawer.app.processEvents()
            continue
        if msg == "update":
            drawer.update_progress(data)
        elif msg == "solution":
            drawer.add_solution(data)
        elif msg == "stop":
            break
        drawer.app.processEvents()


class QtDrawer:
    """Proxy that delegates drawing to a separate process."""

    def __init__(self, cell_size: int = 20, queue_size: int = 5):
        self._queue: Queue = Queue(maxsize=queue_size)
        self._process = Process(
            target=_drawer_process, args=(self._queue, cell_size), daemon=True
        )
        self._process.start()

    def update_progress(self, pic):
        try:
            self._queue.put_nowait(("update", pic.get_pixels().copy()))
        except Full:
            pass

    def add_solution(self, pic):
        try:
            self._queue.put_nowait(("solution", pic.get_pixels().copy()))
        except Full:
            pass

    def close(self):
        try:
            self._queue.put_nowait(("stop", None))
        except Full:
            pass
        self._process.join(timeout=0.2)

    def __del__(self):  # pragma: no cover - defensive cleanup
        if self._process.is_alive():
            self.close()


__all__ = ["QtDrawer"]
