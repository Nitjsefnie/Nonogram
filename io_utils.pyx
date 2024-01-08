from typing import Tuple, List
from numpy import array, ndarray, int8
from picture import Picture

cdef int EMPTY = 0, FULL = 1, UNKNOWN = 2

def load_picture(filename: str) -> Picture:
    with open(filename) as f:
        content: List[str] = [x.strip() for x in f.readlines()]

    result: ndarray = array([[FULL if ch == '#' else
                              EMPTY if ch == '.' else
                              UNKNOWN for ch in line] for line in content], dtype=int8)

    pic: Picture = Picture(result.shape[0], result.shape[1])
    pic[:, :] = result
    return pic

def save_picture(pic: Picture, filename: str):
    with open(filename, 'w') as f:
        for line in pic.get_pixels():
            f.write(''.join('.' if p == EMPTY else '#' if p == FULL else '?' for p in line))
            f.write('\n')

def load_clues(filename: str) -> Tuple[List[List[int]], List[List[int]]]:
    with open(filename) as f:
        content: List[str] = [x.strip() for x in f.readlines()]

    cols: List[List[int]] = []
    rows: List[List[int]] = []
    for line in content:
        if not line:
            cols.append([])
        elif line[0] == '#':
            continue
        elif line == '---':
            rows, cols = cols, rows
        else:
            cols.append([int(x) for x in line.split()])

    return rows, cols

def save_clues(filename: str, row_clues: List[List[int]], col_clues: List[List[int]]):
    with open(filename, 'w') as f:
        for row in row_clues:
            f.write(' '.join(map(str, row)) + '\n')
        f.write('---\n')
        for col in col_clues:
            f.write(' '.join(map(str, col)) + '\n')
