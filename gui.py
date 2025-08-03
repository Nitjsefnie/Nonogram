import sys
from threading import Thread

from PyQt5.QtWidgets import (QApplication, QFileDialog, QPushButton,
                             QVBoxLayout, QWidget)

import solver


class SolveThread(Thread):
    def __init__(self, func, path):
        super().__init__(daemon=True)
        self.func = func
        self.path = path

    def run(self):
        self.func(self.path)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nonogram Solver")

        self.threads = []

        layout = QVBoxLayout(self)

        file_button = QPushButton("Solve File")
        folder_button = QPushButton("Solve Folder")
        layout.addWidget(file_button)
        layout.addWidget(folder_button)

        file_button.clicked.connect(self.solve_file)
        folder_button.clicked.connect(self.solve_folder)

    def solve_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select puzzle file")
        if path:
            thread = SolveThread(solver.solve_file, path)
            self.threads.append(thread)
            thread.start()

    def solve_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            thread = SolveThread(solver.solve_folder, path)
            self.threads.append(thread)
            thread.start()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
