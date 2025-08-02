from numpy import ndarray, array, int16

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
        elif count > 0:
            result.append(count)
            count = 0

    if count:
        result.append(count)
    return array(result, dtype=int16)


def check_line(clue, size):
    length = len(clue)
    summ = 0

    if length == 1:
        return clue[0] <= size

    for i in range(length):
        summ += clue[i]

    return summ + length - 1 <= size
