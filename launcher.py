#!/bin/python3

from tests import *
from sys import argv
from draw import NcursesDrawer


def main():
    drawer = NcursesDrawer()  # Default to terminal
    cpu_count = 100
    args = argv
    i = 1
    drawing = False

    while i < len(args):
        if drawer and not drawing:
            drawer.endwin()
        arg = args[i]
        if arg == "--solve-file":
            if i + 1 < len(args):
                rows, cols = load_clues(args[i + 1])
                solve_file(args[i + 1], drawing=drawing, drawer=drawer)
                i += 2
            else:
                return
        elif arg == "--solve-folder":
            if i + 1 < len(args):
                solve_folder(args[i + 1], drawing=drawing, drawer=drawer)
                i += 2
            else:
                return
        elif arg == "--draw":
            draw_steps(True)
            i += 1
        elif arg == "--draw-results":
            drawing = True
            i += 1
        elif arg == "--no-draw-results":
            drawing = False
            i += 1
        elif arg == "--no-draw":
            draw_steps(False)
            i += 1
        elif arg == "--cpu-count":
            if i + 1 < len(args):
                change_pool_size(int(args[i + 1]))
                i += 2
            else:
                return
        elif arg == "--draw-method":
            if i + 1 < len(args):
                draw_method = args[i + 1]
                if draw_method == "ncurses":
                    drawer = NcursesDrawer()
                elif draw_method == "print":
                    drawer = None
                i += 2
            else:
                return
        else:
            return

    if drawing and drawer is not None:
        drawer.wait_for_quit()
        drawer.endwin()


if __name__ == "__main__":
    main()
    terminate_pool()
