from string_builder import StringBuilder
from picture import Picture
from globalvars import Global
from math import log10
from itertools import islice
from os import system

cdef int EMPTY = 0, FULL = 1, UNKNOWN = 2

cpdef void draw_steps(bint drawing):
    Global.drawing = drawing


cdef void draw_row(row, builder):
    printing_pattern = { EMPTY: "⬜", FULL: "⬛", UNKNOWN: "ｘ" }
    builder.add(''.join(draw_iter(row, printing_pattern)), end='')


def draw_iter(row, patt):
    yield from (patt[el] for el in row)


def draw_col_legend(nonogram, pic=None):
    cdef dict x
    cdef list col_legends
    cdef int col, digits, digits_tcc, i, j
    cdef str col_legend

    x = {"0": "０", "1": "１", "2": "２", "3": "３", "4": "４", "5": "５", "6": "６", "7": "７", "8": "８", "9": "９" , " ": "　"}
    col_legends = [None for _ in nonogram[0]]
    digits = int(log10(len(nonogram[0]))) + 1

    if pic is not None:
        digits_tcc = int(log10(max(pic.tcc))) + 1

    for col in range(len(nonogram[0])):
        if pic is not None:
            col_legends[col] = ' '.join([str(pic.tcc[col]).rjust(digits_tcc), str(col).rjust(digits)])
        else:
            col_legends[col] = str(col).rjust(digits)

    return col_legends, x


cpdef draw(nonogram, backtrack_progress=None, pic=None):
    cdef int digits, digits_trc, i
    cdef object it

    if backtrack_progress is None:
        backtrack_progress = []

    digits = int(log10(len(nonogram))) + 1
    builder = StringBuilder()


    col_legends, x = draw_col_legend(nonogram, pic)
    if pic is not None:
        digits_trc = int(log10(max(pic.trc))) + 1
        builder.add("\033[2J\033[H")

    for i in range(len(col_legends[0])):
        for j in range(len(col_legends)):
            builder.add(x[col_legends[j][i]], end='')
        builder.add()


    for i, row in enumerate(nonogram):
        draw_row(row, builder)
        if not backtrack_progress:
            if pic is not None:
                builder.add(str(i).rjust(digits), str(pic.trc[i]).rjust(digits_trc))
            else:
                builder.add(str(i).rjust(digits))
        else:
            builder.add(str(i).rjust(digits), str(pic.trc[i]).rjust(digits_trc), end=" ")
            if i == 0:
                it = every_second(backtrack_progress)
                builder.add("backtrack progress:")
            else:
                builder.add(' | '.join(islice(it, 13 - pic.width // 40)))
    builder.add()
    print(builder.build())


def every_second(lst):
    p = 0
    q = 1
    for i in range(0, len(lst), 3):
        yield lst[i]
    for i in range(2, len(lst), 3):
        p *= lst[i]
        q *= lst[i]
        p += lst[i - 1]
    yield f"{(p/q):5%}"
