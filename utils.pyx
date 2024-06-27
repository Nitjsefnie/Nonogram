from numpy import ndarray, array, int16
from libc.stdint cimport int16_t

cdef int EMPTY = 0, FULL = 1, UNKNOWN = 2

cpdef int16_t[:] gen_line_clues(line):
    if line.size == 0:
        return array([], dtype=int16)
    cdef list result = []
    cdef int count = 0
    cdef int val
    for val in line:
        if val == FULL:
            count += 1
        elif count > 0:
            result.append(count)
            count = 0

    if count:
        result.append(count)
    return array(result, dtype=int16)


cpdef bint check_line(clue, size):
    cdef int length = len(clue)
    cdef int summ = 0
    cdef int i

    if length == 1:
        return clue[0] <= size

    for i in range(length):
        summ += clue[i]

    return summ + length - 1 <= size
