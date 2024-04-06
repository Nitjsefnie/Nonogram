from tests import *
from sys import argv

def main():
    cpu_count = 100
    args = argv
    i = 1
    while i < len(args):
        arg = args[i]
        if arg == "--solve-file":
            if i + 1 < len(args):
                solve_file(args[i + 1])
                i += 2
            else:
                return
        elif arg == "--solve-folder":
            if i + 1 < len(args):
                solve_folder(args[i + 1])
                i += 2
            else:
                return
        elif arg == "--draw":
            draw_steps(True)
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
        else:
            return

    # Apply CPU count change and terminate the pool
    terminate_pool()

if __name__ == "__main__":
    main()
