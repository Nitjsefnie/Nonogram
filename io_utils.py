from functools import lru_cache
from numpy import array, int8
from picture import Picture

EMPTY = 0
FULL = 1
UNKNOWN = 2


def load_picture(filename):
    with open(filename) as f:
        result = array([[FULL if ch == '#' else
                         EMPTY if ch == '.' else
                         UNKNOWN for ch in line.strip()] for line in f], dtype=int8)

    pic = Picture(result.shape[0], result.shape[1])
    pic[:, :] = result
    return pic


def save_picture(pic, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(''.join('.' if p == EMPTY else '#' if p == FULL else '?' for p in line)
                          for line in pic.get_pixels()))
        f.write('\n')


@lru_cache(maxsize=1024)
def load_clues(filename):
    with open(filename) as f:
        content = [x.strip() for x in f]

    cols = []
    rows = []
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


def save_clues(filename, row_clues, col_clues):
    with open(filename, 'w') as f:
        f.write('\n'.join(' '.join(map(str, row)) for row in row_clues))
        f.write('\n---\n')
        f.write('\n'.join(' '.join(map(str, col)) for col in col_clues))
        f.write('\n')
