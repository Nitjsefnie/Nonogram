import pool
from utils import check_line, gen_line_clues
from picture import Picture
from globalvars import Global
from pool import change_pool_size, terminate_pool, reset_pool
from draw_utils import draw, draw_steps
from io_utils import *

from time import time, sleep
from os import listdir, getenv, rename
from os.path import isfile, join
from sys import setrecursionlimit
from math import log10
from itertools import chain
from numpy import full, int64, ndenumerate, iinfo
from functools import lru_cache, wraps

setrecursionlimit(10000)

Pixel = int
EMPTY, FULL, UNKNOWN = 0, 1, 2

start_time = time()

def clues_valid(rows, cols) -> bool:
    cdef int height = len(rows)
    cdef int width = len(cols)
    cdef int row_sum = 0
    cdef int col_sum = 0
    cdef list row
    cdef list col

    for row in rows:
        row_sum += sum(row)
        if not check_line(row, width):
            return False

    for col in cols:
        col_sum += sum(col)
        if not check_line(col, height):
            return False

    return row_sum == col_sum

def conditional_lru_cache(func):
    cached_func = lru_cache(maxsize=2 ** 20)(func)

    @wraps(func)
    def wrapper(arg1, arg2 = None):
        if getenv('IS_CHILD_PROCESS') == '1':
            return func(arg1, arg2)
        else:
            return cached_func(arg1.tobytes() if isinstance(arg1, ndarray) else tuple(arg1), arg2)

    def cache_info():
        if getenv('IS_CHILD_PROCESS') != '1':
            return cached_func.cache_info()

    def cache_clear():
        if getenv('IS_CHILD_PROCESS') != '1':
            cached_func.cache_clear()

    wrapper.cache_info = cache_info
    wrapper.cache_clear = cache_clear

    return wrapper

@conditional_lru_cache
def len_gen_lines(line, states):
    cdef int state, val, state_next
    cdef int len_states = len(states)
    cdef dict states_dict = {0: 1}
    cdef dict new_dict

    for val in line:
        new_dict = {}
        for state, count in states_dict.items():
            state_next = state + 1
            if val == UNKNOWN:
                if state_next < len_states:
                    new_dict[state_next] = new_dict.get(state_next, 0) + count
                if states[state] == EMPTY:
                    new_dict[state] = new_dict.get(state, 0) + count
            elif val == EMPTY:
                if state_next < len_states and states[state_next] == EMPTY:
                    new_dict[state_next] = new_dict.get(state_next, 0) + count
                if states[state] == EMPTY:
                    new_dict[state] = new_dict.get(state, 0) + count
            elif val == FULL and state_next < len_states and states[state_next] == FULL:
                new_dict[state_next] = new_dict.get(state_next, 0) + count

        states_dict = new_dict

    return states_dict.get(len_states - 1, 0) + states_dict.get(len_states - 2, 0)

@conditional_lru_cache
def states_pregen(clue, arg2 = None):
    cdef list states = [0]
    for nr in clue:
        states.extend([1] * nr)
        states.append(0)
    return tuple(states)

def solve(rows, cols, cheated_pixels=[], drawer=None, lookahead=0):

    len_gen_lines.cache_clear()
    pic = Picture(len(rows), len(cols))

    for row, col, val in cheated_pixels:
        pic.set_pixel(row, col, val)

    mapped_rows = [(i, states_pregen(x)) for i, x in enumerate(rows)]
    mapped_cols = [(i, states_pregen(x)) for i, x in enumerate(cols)]

    yield from solve_real(mapped_rows, mapped_cols, pic, 0, [], drawer=drawer, correct_pixels=True, lookahead=lookahead)

def solve_one(clue, index, is_col, pic):
    line = pic.get_col(index) if is_col else pic.get_row(index, copying=False)
    res_size = len_gen_lines(line, clue)
    result = []

    if UNKNOWN not in line:
        if is_col:
            pic.set_col_solved(index)
        else:
            pic.set_row_solved(index)
        return True, []

    for i, val in enumerate(line):
        if val != UNKNOWN:
            continue
        line[i] = FULL
        full_res_size = len_gen_lines(line, clue)
        if res_size == full_res_size:
            result.append((i if is_col else index, index if is_col else i, FULL))
        elif full_res_size == 0:
            result.append((i if is_col else index, index if is_col else i, EMPTY))
        line[i] = UNKNOWN

    return True, result

def write_intersection(lst, pic, is_row, correct_pixels=False):
    cdef int i, j, val
    for i, j, val in lst:
        if pic.get_pixel(i, j) == UNKNOWN:
            pic.set_pixel(i, j, val)
            if correct_pixels:
                pic.correct_pixels[i, j] = True
            if is_row:
                pic.cols_to_solve[j] = True
            else:
                pic.rows_to_solve[i] = True

def draw_and_check(pic, mapped_rows, mapped_cols, back_progress, depth, drawer):
    row_complexities = [pic.get_row_complexity(i) for i, _ in mapped_rows]
    col_complexities = [pic.get_col_complexity(i) for i, _ in mapped_cols]

    if Global.drawing:
        if drawer:
            drawer.draw_nonogram(pic.get_pixels(), row_complexities, col_complexities, back_progress)
        else:
            draw(pic.get_pixels(), back_progress, pic)
            print(len_gen_lines.cache_info())

    if not solve_check(pic, mapped_rows, mapped_cols, depth != 0):
        return False

    return True

def solve_real(mapped_rows, mapped_cols, pic, depth, back_progress, contradicting=False, drawer=None, correct_pixels=False, lookahead=0):
    global start_time
    if time() - start_time > 60:
        start_time = time()
        reset_pool()
    if not draw_and_check(pic, mapped_rows, mapped_cols, back_progress, depth, drawer):
        return

    if pic.is_solved():
        if not Global.stored_backtrack:
            Global.stored_backtrack = back_progress
        yield pic
        return

    if not solve_rows_or_cols(mapped_rows if depth % 2 != 0 else mapped_cols, pic, depth % 2 != 0, depth, correct_pixels=correct_pixels):
        return

    if not any(pic.rows_to_solve) and not any(pic.cols_to_solve):
        if contradicting:
            yield pic
            return
        yield from solve_contra(mapped_rows, mapped_cols, pic, depth, back_progress, drawer=drawer, correct_pixels=correct_pixels, max_lookahead=lookahead)
        return

    yield from solve_real(mapped_rows, mapped_cols, pic, depth + 1, back_progress, contradicting, drawer=drawer, correct_pixels=correct_pixels, lookahead=lookahead)

def solve_rows_or_cols(mapped, pic, is_row, depth, correct_pixels=False):
    solve_these = [
        (clue, i, not is_row, pic) for i, clue in mapped
        if (is_row and pic.rows_to_solve[i]) or (not is_row and pic.cols_to_solve[i])
    ]

    for i, clue in mapped:
        if (is_row and pic.rows_to_solve[i]) or (not is_row and pic.cols_to_solve[i]):
            if is_row:
                pic.rows_to_solve[i] = False
            else:
                pic.cols_to_solve[i] = False

    res = Global.pool.starmap(solve_one, solve_these)

    for boo, pix_loc in res:
        if not boo:
            return False
        write_intersection(pix_loc, pic, is_row, correct_pixels=correct_pixels)

    return True

def recalculate_pixel_complexities(mapped_rows, mapped_cols, pic):
    rows = filter(lambda x: pic.get_row_complexity(x[0]) > 1, map(lambda x: (x[0], x[1], True), mapped_rows))
    cols = filter(lambda x: pic.get_col_complexity(x[0]) > 1, map(lambda x: (x[0], x[1], False), mapped_cols))
    chained = chain(rows, cols)

    for index, clue, is_row in chained:
        if (is_row and not pic.row_changed_complexity(index)) or (not is_row and not pic.col_changed_complexity(index)):
            continue
        line = pic.get_row(index, copying=False) if is_row else pic.get_col(index)
        complexity = pic.get_row_complexity(index) if is_row else pic.get_col_complexity(index)

        for i, val in enumerate(line):
            if val != UNKNOWN:
                continue
            line[i] = EMPTY
            new_zero = len_gen_lines(line, clue)
            new_one = complexity - new_zero
            line[i] = UNKNOWN
            pic.set_pixel_complexity(
                index if is_row else i,
                i if is_row else index,
                zero_complexity=new_zero,
                one_complexity=new_one
            )

    pic.store_current_complexities()


def solve_contra(mapped_rows, mapped_cols, pic, depth, back_progress, drawer=None, max_lookahead=0, lookaheading=False, correct_pixels=False):
    filled = pic.count_matching_pixels(lambda x: x != UNKNOWN)
    current_lookahead = 0
    while current_lookahead < max_lookahead:
        iterator = solve_contra_real(mapped_rows, mapped_cols, pic, depth, back_progress, drawer=drawer, lookahead=current_lookahead, max_lookahead=max_lookahead)
        curr_pic = next(iterator, None)
        while curr_pic is not None:
            if curr_pic.is_solved():
                yield curr_pic
            else:
                break
            curr_pic = next(iterator, None)
        if curr_pic is not None and curr_pic.is_solved():
            return
        pic = curr_pic
        if pic is None:
            return
        new_filled = pic.count_matching_pixels(lambda x: x != UNKNOWN)
        if new_filled != filled:
            filled = new_filled
            current_lookahead = 0
            continue
        current_lookahead += 1
    yield from solve_contra_real(mapped_rows, mapped_cols, pic, depth, back_progress, drawer=drawer, lookahead=current_lookahead, max_lookahead=max_lookahead)



def solve_contra_real(mapped_rows, mapped_cols, pic, depth, back_progress, min_neighs=4, sorted_list=None, tested=None, drawer=None, lookahead=0, lookaheading=False, max_lookahead=0):
    if min_neighs == 4 and not lookaheading:
        recalculate_pixel_complexities(mapped_rows, mapped_cols, pic)
        indexed = (
            (value, (index[0], index[1]), index[2])
            for index, value in ndenumerate(pic.pixel_complexity) if value != iinfo(int64).max
        )
        sorted_list = sorted(indexed, key=lambda x: (-(pic.pixel_gain[x[1][0], x[1][1], x[2]] + 1), x[0]))
        tested = full((pic.height, pic.width, 2), False, dtype=bool)

    rjust_val_rows = 1 + int(log10(pic.height))
    rjust_val_cols = 1 + int(log10(pic.width))
    len_sorted = len(sorted_list)
    pic_size = pic.get_pixels().size
    new_filled = filled = pic.count_matching_pixels(lambda x: x != UNKNOWN)
    sorted_list = sorted(sorted_list, key=lambda x: (-(pic.pixel_gain[x[1][0], x[1][1], x[2]] + 1), x[0]))

    for i, val in enumerate(sorted_list):
        comp, index, value = val

        if pic.get_pixel(index) != UNKNOWN or pic.count_neighbours(index) < min_neighs or tested[index[0], index[1], value]:
            continue

        tested[index[0], index[1], value] = True
        pic2 = pic.copy_pic(small=(False if lookahead > 0 else True))
        pic2.set_pixel(index, value)
        pic2.set_should_solve_row(index[0], True)
        pic2.set_should_solve_col(index[1], True)
        new_back_progress = back_progress + [
            f"{pic.count_neighbours(index)}r{index[0]:>{rjust_val_rows}}c{index[1]:>{rjust_val_cols}}: {i / len_sorted:>3.0%}",
            i,
            len_sorted
        ]
        pic2 = next(solve_real(mapped_rows, mapped_cols, pic2, depth, new_back_progress, True, drawer=drawer), None)

        if pic2 is not None:
            if not pic2.is_solved():
                pic.pixel_gain[index[0], index[1], value] = pic2.count_matching_pixels(lambda x: x != UNKNOWN) - new_filled
                for diff in (pic2 - pic):
                    tested[diff] = True
                if lookahead > 0:
                    pic3 = next(solve_contra_real(mapped_rows, mapped_cols, pic2, depth, new_back_progress, 4, sorted_list, tested.copy(), drawer, lookahead - 1, True, max_lookahead), None)
                    # if pic2 has no solutions
                    if pic3 is None:
                        # set opposite pixel
                        pic2 = None
                        pic.set_pixel(index, value ^ 1)
                        pic.set_should_solve_row(index[0], True)
                        pic.set_should_solve_col(index[1], True)
                        pic = next(solve_real(mapped_rows, mapped_cols, pic, depth, back_progress, True, drawer=drawer), None)

                        # even that has no solution
                        if pic is None:
                            return

                        # solution was actually found
                        if pic.is_solved():
                            yield pic
                            return

                        new_filled = pic.count_matching_pixels(lambda x: x != UNKNOWN)

                        if min_neighs <= 2 and filled + pic_size / 25 <= new_filled:
                            break
                continue
            else:
                yield pic2

        pic2 = None
        pic.set_pixel(index, value ^ 1)
        pic.set_should_solve_row(index[0], True)
        pic.set_should_solve_col(index[1], True)
        pic = next(solve_real(mapped_rows, mapped_cols, pic, depth, back_progress, True, drawer=drawer), None)

        if pic is None:
            return

        if pic.is_solved():
            yield pic
            return

        new_filled = pic.count_matching_pixels(lambda x: x != UNKNOWN)

        if (min_neighs <= 2 and filled + pic_size / 25 <= new_filled) or (lookaheading and new_filled != filled):
            break

    if filled == new_filled:
        if min_neighs == 0:
            tested = None
            if lookaheading:
                yield pic
                return
            if lookahead < max_lookahead:
                yield pic
                return
            yield from solve_backtrack(mapped_rows, mapped_cols, pic, depth, back_progress, sorted_list, drawer=drawer, lookahead=max_lookahead)
            return
        yield from solve_contra_real(mapped_rows, mapped_cols, pic, depth, back_progress, min_neighs - 1, sorted_list, tested, drawer, lookahead, lookaheading, max_lookahead)
        return

    sorted_list = None
    tested = None
    if lookaheading:
        yield pic
        return
    yield from solve_real(mapped_rows, mapped_cols, pic, depth + 1, back_progress, drawer=drawer, lookahead=lookahead)


def solve_backtrack(mapped_rows, mapped_cols, pic, depth, back_progress, sorted_list, drawer=None, lookahead=0):
    sorted_list = sorted(sorted_list, key=lambda x: -(pic.pixel_gain[x[1][0], x[1][1], x[2]] + 1))

    for i, val in enumerate(sorted_list):
        comp, index, value = val
        if pic.get_pixel(index) != UNKNOWN:
            continue
        break

    rjust_val_rows = 1 + int(log10(pic.height))
    rjust_val_cols = 1 + int(log10(pic.width))
    new_sols = [value, value ^ 1]
    row, col = index

    for k, val in enumerate(new_sols):
        pic2 = pic.copy_pic()
        pic2.set_pixel(index, val)
        pic2.cols_to_solve[col] = True
        pic2.rows_to_solve[row] = True
        yield from solve_real(mapped_rows, mapped_cols, pic2, depth, back_progress + [f" r{row:>{rjust_val_rows}d}c{col:>{rjust_val_cols}d}: {f'{k / 2:.0%}':>3}", k, 2], drawer=drawer, lookahead=lookahead)


def solve_check(pic, mapped_rows, mapped_cols, total=False) -> bool:
    for i, clue in mapped_rows:
        if not total and not pic.rows_to_solve[i]:
            continue
        if i in pic.solved_rows:
            continue
        line = pic.get_row(i)
        complexity = len_gen_lines(line, clue)
        if complexity == 0:
            return False
        if complexity == 1 and UNKNOWN not in line:
            pic.solved_rows.add(i)
        pic.set_row_complexity(i, complexity)

    for i, clue in mapped_cols:
        if not pic.cols_to_solve[i]:
            continue
        if i in pic.solved_cols:
            continue
        line = pic.get_col(i)
        complexity = len_gen_lines(line, clue)
        if complexity == 0:
            return False
        if complexity == 1 and UNKNOWN not in line:
            pic.solved_cols.add(i)
        pic.set_col_complexity(i, complexity)

    return True


def solve_sort_all() -> None:
    for dir_loc in [
        "demo_nonograms/easiest",
        "demo_nonograms/easy",
        "demo_nonograms/medium",
        "demo_nonograms/hard",
        "demo_nonograms/extreme",
    ]:
        only_files = sorted(join(dir_loc, f) for f in listdir(dir_loc) if isfile(join(dir_loc, f)))

        for i, file in enumerate(only_files):
            lowest = None
            for _ in range(5):
                reset_pool()
                start = time()
                solve_file(file, drawer=None, lookahead=0)
                solve_time = time() - start
                if lowest is None or lowest > solve_time:
                    lowest = solve_time
            folder = "easiest" if lowest < 0.1 else "easy" if lowest < 5 else "medium" if lowest < 30 else "hard" if lowest < 600 else "extreme" if lowest < 21600 else "insane"
            new_path = join("demo_nonograms", folder, file.split("/")[file.count("/")])
            if folder not in file:
                print(f"moving {file} to {new_path}")
                rename(file, new_path)
            print()

    # print(f"\nAll from {loc} on {Global.pool_size} threads: {time() - start}\n")


def solve_folder(loc, drawing=False, drawer=None, lookahead=0) -> None:
    start = time()
    only_files = sorted(join(loc, f) for f in listdir(loc) if isfile(join(loc, f)))

    for file in only_files:
        reset_pool()
        solve_file(file, drawing=drawing, drawer=drawer, lookahead=lookahead)

    print(f"\nAll from {loc} on {Global.pool_size} threads: {time() - start}\n")


def solve_file(location, drawing=False, cheated_pixels=[], number=-1, drawer=None, lookahead=0):
    rows, cols = load_clues(location)
    if not clues_valid(rows, cols):
        print(f"Invalid clues: {location}")
        return
    start = time()
    i = 0
    for pic in solve(rows, cols, cheated_pixels=cheated_pixels, drawer=drawer, lookahead=lookahead):
        if drawing and not drawer:
            draw(pic.get_pixels())
        i += 1
        if drawing:
            print(f"Solution {i}")
            draw(pic.get_pixels())
        if i == number:
            break
    print(f"{location} on {Global.pool_size} threads: {time() - start}, found {i} solutions")

change_pool_size()