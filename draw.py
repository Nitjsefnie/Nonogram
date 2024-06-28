import curses
from itertools import islice
from math import log10


def every_second(lst):
    p = 0
    q = 1
    for i in range(0, len(lst), 3):
        yield lst[i]
    for i in range(2, len(lst), 3):
        p *= lst[i]
        q *= lst[i]
        p += lst[i - 1]
    yield f"{(p / q):5%}"


class NcursesDrawer:
    def __init__(self):
        self.stdscr = curses.initscr()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Full cell
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Empty cell
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Unknown cell
        self.cell_size = 2  # Adjust cell size if needed
        self.previous_frame = None  # Buffer to store the previous frame
        self.max_backtrack_offset_x = 1
        self.max_row_comp_offset = 1
        self.max_col_comp_length = 1

    def draw_nonogram(self, nonogram, row_complexities, col_complexities, back_progress):
        self._draw_on_ncurses(nonogram, row_complexities, col_complexities, back_progress)

    def _draw_on_ncurses(self, nonogram, row_complexities, col_complexities, back_progress):
        rows = len(nonogram)
        cols = len(nonogram[0])

        # Get the dimensions of the terminal window
        max_y, max_x = self.stdscr.getmaxyx()

        if self.previous_frame is None:
            self.previous_frame = [[None] * cols for _ in range(rows)]

        # Calculate the vertical offset for column indexes
        col_index_height = int(log10(cols)) + 1  # Adjust as needed for the number of index lines
        grid_start_y = col_index_height + 2

        # Calculate the horizontal offset for backtrack progress
        self.max_row_comp_offset = complexity_shift = max(2 + int(log10(max(row_complexities))), self.max_row_comp_offset)
        self.max_backtrack_offset_x = backtrack_offset_x = max(cols * self.cell_size + 9 + complexity_shift,  self.max_backtrack_offset_x)

        # Draw column indexes vertically at the top
        for j in range(cols):
            col_str = f"{j:{col_index_height}}"
            for k, char in enumerate(col_str):
                if (k + 1) < max_y and (j * self.cell_size + 6) < max_x:  # Adjusted position to prevent overlap
                    self.stdscr.addstr(k + 1, j * self.cell_size + 6, char)

        # Draw the grid with borders and row indexes
        for i, row in enumerate(nonogram):
            if (i + grid_start_y) < max_y:
                self.stdscr.addstr(i + grid_start_y, 2, f"{i:2}")
            for j, cell in enumerate(row):
                x = j * self.cell_size + 6
                y = i + grid_start_y
                if y < max_y and x < max_x:
                    if cell == 1:
                        color_pair = curses.color_pair(1)
                        symbol = '  '
                    elif cell == 0:
                        color_pair = curses.color_pair(2)
                        symbol = '  '
                    else:
                        color_pair = curses.color_pair(3)
                        symbol = '  '
                    if self.previous_frame[i][j] != cell:
                        self.previous_frame[i][j] = cell
                        self.stdscr.addstr(y, x, symbol, color_pair)

        # Draw row complexities at the end of each row
        for i, complexity in enumerate(row_complexities):
            if (i + grid_start_y) < max_y and (cols * self.cell_size + 8 + complexity_shift) < max_x:
                self.stdscr.addstr(i + grid_start_y, cols * self.cell_size + 8, f"{complexity:<{complexity_shift}}")

        # Draw column complexities at the bottom
        self.max_col_comp_length = col_complexity_length = max(1 + int(log10(max(col_complexities))), self.max_col_comp_length)
        for i, complexity in enumerate(col_complexities):
            complexity_str = f"{complexity:<{col_complexity_length}}"
            for k, char in enumerate(complexity_str):
                if (rows + grid_start_y + k + 1) < max_y and (i * self.cell_size + 6) < max_x:
                    self.stdscr.addstr(rows + grid_start_y + k + 1, i * self.cell_size + 6, char)

        # Draw borders
        for y in range(grid_start_y, grid_start_y + rows + 1):
            self.stdscr.addstr(y, 4, '|')
            self.stdscr.addstr(y, cols * self.cell_size + 6, '|')

        for x in range(4, cols * self.cell_size + 7, 2):
            self.stdscr.addstr(grid_start_y - 1, x, '-')
            self.stdscr.addstr(grid_start_y + rows, x, '-')

        # Clear the backtrack progress area
        for y in range(grid_start_y, max_y):
            self.stdscr.addstr(y, self.max_backtrack_offset_x, ' ' * (max_x - self.max_backtrack_offset_x - 1))

        # Draw backtrack progress
        if back_progress:
            it = every_second(back_progress)
            builder = "backtrack progress:"
            self.stdscr.addstr(grid_start_y, self.max_backtrack_offset_x, builder[:max_x - self.max_backtrack_offset_x])

            # Initialize positions for printing backtrack progress
            y_pos = grid_start_y + 1
            x_pos = self.max_backtrack_offset_x
            max_width = max_x - self.max_backtrack_offset_x

            progress = next(it, None)
            while progress is not None:
            #for progress in it:
                progress_str = str(progress)
                if x_pos + len(progress_str) + 3 > max_width:  # +3 for the " | " separator
                    y_pos += 1
                    x_pos = self.max_backtrack_offset_x
                if y_pos < max_y:
                    self.stdscr.addstr(y_pos, x_pos, progress_str)
                    x_pos += len(progress_str) + 3
                    progress = next(it, None)
                    if progress and x_pos + 3 <= max_width:  # Add separator if there is space
                        self.stdscr.addstr(y_pos, x_pos - 3, " | ")

        self.stdscr.refresh()

    def wait_for_quit(self):
        while True:
            key = self.stdscr.getch()
            if key == ord('q'):
                break

    def endwin(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
