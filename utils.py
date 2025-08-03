from numpy import array, int16

EMPTY = 0
FULL = 1
UNKNOWN = 2


def gen_line_clues(line):
    if line.size == 0:
        return array([], dtype=int16)
    result = []
    count = 0
    for val in line:
        if val == FULL:
            count += 1
        elif count:
            result.append(count)
            count = 0
    if count:
        result.append(count)
    return array(result, dtype=int16)


def check_line(clue, size):
    if len(clue) == 1:
        return clue[0] <= size
    return sum(clue) + len(clue) - 1 <= size
