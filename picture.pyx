from numpy import ndarray, full, int8, int32, int64, count_nonzero, iinfo, where
from typing import Optional, Tuple, Union, Callable

cdef int EMPTY = 0, FULL = 1, UNKNOWN = 2

class Picture:
    __slots__ = ('height', 'width', '__pixels',
                 'solved_rows', 'solved_cols', 'rows_to_solve', 'cols_to_solve',
                 'trc', 'tcc', 'old_trc', 'old_tcc',
                 'pixel_complexity', 'pixel_gain', 'correct_pixels')

    def __init__(self, int height, int width, *, bint generating=True):
        self.height: int = height
        self.width: int = width

        if generating:
            self.__pixels: ndarray = full((height, width), UNKNOWN, dtype=int8)
            self.correct_pixels = full((height, width), False, dtype=bool)

            self.solved_rows: set[int] = set()
            self.solved_cols: set[int] = set()

            self.rows_to_solve: ndarray = full(height, True, dtype=bool)
            self.cols_to_solve: ndarray = full(width, True, dtype=bool)

            self.trc: ndarray = full(height, 1, dtype=int64)
            self.tcc: ndarray = full(width, 1, dtype=int64)

            self.pixel_complexity: ndarray = full((height, width, 2), iinfo(int64).max, dtype=int64)

            self.pixel_gain: ndarray = full((height, width, 2), iinfo(int32).min, dtype=int32)

            self.old_trc: ndarray = full(height, iinfo(int64).max, dtype=int64)
            self.old_tcc: ndarray = full(width, iinfo(int64).max, dtype=int64)

        else:
            self.__pixels: Optional[ndarray] = None
            self.correct_pixels = None

            self.solved_rows: Optional[set[int]] = None
            self.solved_cols: Optional[set[int]] = None

            self.rows_to_solve: Optional[ndarray] = None
            self.cols_to_solve: Optional[ndarray] = None

            self.trc: Optional[ndarray] = None
            self.tcc: Optional[ndarray] = None

            self.pixel_complexity: Optional[ndarray] = None

            self.pixel_gain: Optional[ndarray] = None

            self.old_trc: Optional[ndarray] = None
            self.old_tcc: Optional[ndarray] = None


    def __bool__(self):
        raise ValueError("Boolean evaluation is not allowed")

    def get_row(self, int row, bint copying=False) -> ndarray:
        if copying:
            return self.__pixels[row].copy()
        return self.__pixels[row]

    def get_col(self, int col, bint copying=False) -> ndarray:
        if copying:
            return self.__pixels[:, col]
        return self.__pixels[:, col]

    def get_row_complexity(self, int row) -> int64:
        return self.trc[row]

    def get_col_complexity(self, int col) -> int64:
        return self.tcc[col]

    def set_row_complexity(self, int row, complexity: int64):
        self.trc[row] = complexity

    def set_col_complexity(self, int col, complexity: int64):
        self.tcc[col] = complexity

    def is_row_solved(self, int row) -> bool:
        if row in self.solved_rows:
            return True
        if not (UNKNOWN in self.__pixels[row]):
            self.solved_rows.add(row)
            return True
        return False

    def is_col_solved(self, int col) -> bool:
        if col in self.solved_cols:
            return True
        if not (UNKNOWN in self.__pixels[:, col]):
            self.solved_cols.add(col)
            return True
        return False

    def set_row_solved(self, int row):
        self.solved_rows.add(row)

    def set_col_solved(self, int col):
        self.solved_cols.add(col)

    def should_solve_row(self, int row) -> bool:
        return self.rows_to_solve[row]

    def should_solve_col(self, int col) -> bool:
        return self.cols_to_solve[col]

    def set_should_solve_row(self, int row, bint val) -> bool:
        self.rows_to_solve[row] = val

    def set_should_solve_col(self, int col, bint val) -> bool:
        self.cols_to_solve[col] = val

    def copy_pic(self, *, small = False) -> 'Picture':
        pic2: Picture = Picture(self.height, self.width, generating = False)
        pic2.__pixels = self.__pixels.copy()
        pic2.solved_rows = set(self.solved_rows)
        pic2.solved_cols = set(self.solved_cols)
        pic2.rows_to_solve = self.rows_to_solve.copy()
        pic2.cols_to_solve = self.cols_to_solve.copy()
        pic2.trc = self.trc.copy()
        pic2.tcc = self.tcc.copy()
        pic2.correct_pixels = self.correct_pixels
        if not small:
            pic2.pixel_complexity = self.pixel_complexity.copy()
            pic2.pixel_gain = self.pixel_gain.copy()
            pic2.old_trc = self.old_trc.copy()
            pic2.old_tcc = self.old_tcc.copy()
        return pic2

    def is_solved(self) -> bool:
        return not (UNKNOWN in self.__pixels)

    def get_pixel_complexity(self, row_col: Union[Tuple[int, int], int], col: Optional[int] = None) -> Tuple[int64, int64]:
        if isinstance(row_col, tuple):
            row, col = row_col
        else:
            row = row_col
        return self.pixel_complexity[row, col]

    def get_pixel(self, row_col: Union[Tuple[int, int], int], col: Optional[int] = None) -> int8:
        if isinstance(row_col, tuple):
            row, col = row_col
        else:
            row = row_col
        return self.__pixels[row, col]

    def set_pixel(self, row_col: Union[Tuple[int, int], int], int col, value: Optional[int8] = None):
        if isinstance(row_col, tuple):
            value = col
            row, col = row_col
        else:
            row = row_col
        self.__pixels[row, col] = value

    def set_pixel_complexity(self, row_col: Union[Tuple[int, int], int], col: Optional[int] = None,
                             *, zero_complexity: Optional[int64] = None, one_complexity: Optional[int64] = None):
        if isinstance(row_col, tuple):
            row, col = row_col
        else:
            row = row_col

        if zero_complexity is not None and zero_complexity < self.pixel_complexity[row, col, 0]:
            self.pixel_complexity[row, col, 0] = zero_complexity
        if one_complexity is not None and one_complexity < self.pixel_complexity[row, col, 1]:
            self.pixel_complexity[row, col, 1] = one_complexity

    def get_pixels(self) -> ndarray:
        return self.__pixels

    def count_matching_pixels(self, predicate: Callable[[int], bool]) -> int:
        return count_nonzero(predicate(self.__pixels))

    def count_neighbours(self, row_col: Union[Tuple[int, int], int], col: Optional[int] = None) -> int:
        if isinstance(row_col, tuple):
            row, col = row_col
        else:
            row = row_col

        height, width = self.height, self.width
        count = 0
        if row == 0 or self.__pixels[row - 1, col] != UNKNOWN:
            count += 1
        if row == height - 1 or self.__pixels[row + 1, col] != UNKNOWN:
            count += 1
        if col == 0 or self.__pixels[row, col -  1] != UNKNOWN:
            count += 1
        if col == width - 1 or self.__pixels[row, col + 1] != UNKNOWN:
            count += 1

        return count

    def store_current_complexities(self):
        self.old_trc = self.trc.copy()
        self.old_tcc = self.tcc.copy()

    def row_changed_complexity(self, int index) -> bool:
        return self.trc[index] != self.old_trc[index]

    def col_changed_complexity(self, int index) -> bool:
        return self.tcc[index] != self.old_tcc[index]

    def __getitem__(self, key):
        return self.__pixels[key]

    def __setitem__(self, key, value):
        self.__pixels[key] = value

    def __eq__(self, obj):
        if isinstance(obj, Picture):
            return (self.get_pixels() == obj.get_pixels()).all()
        return False

    def __sub__(self, obj):
        if not isinstance(obj, Picture):
            raise TypeError
        if self == obj:
            return []

        result = []
        pixs, pixs2 = self.get_pixels(), obj.get_pixels()
        indexes = where(pixs != pixs2)
        differing_values = pixs[indexes]
        return [(idx[0], idx[1], val) for idx, val in zip(zip(*indexes), differing_values)]

    def set_all_pixels(self, pixels):
        self.__pixels = pixels
