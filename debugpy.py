#cython: profile=True, language_level=3, linetrace=True, boundscheck=False, cdivision=True
from time import time
from os import remove, listdir, cpu_count
from os.path import isfile, join
from typing import Tuple, List, Set, Optional
from math import comb
from multiprocessing import Pool

Pixel = int
EMPTY, FULL, UNKNOWN = 0, 1, 2

Clue = List[int]
Nonogram = List[List[int]]
pool_size = cpu_count()


class Picture:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.pix_num = width * height
        self.pix_solved = 0
        self.pixels = [[UNKNOWN for _ in range(width)] for _ in range(height)]
        self.solved_rows: Set[int] = set()
        self.solved_cols: Set[int] = set()
        self.rows_to_solve: Set[int] = set()
        self.cols_to_solve: Set[int] = set()
        self.trc = [ 0 ] * height
        self.tcc = [ 0 ] * width

    def get_column(self, column: int) -> List[int]:
        return [self.pixels[index][column]
                for index, _ in enumerate(self.pixels)]

    def copy_pic(self) -> 'Picture':
        pic2 = Picture(self.height, self.width)
        pic2.solved_rows = set(self.solved_rows)
        pic2.solved_cols = set(self.solved_cols)
        pic2.rows_to_solve = set()
        pic2.pixels = [x[:] for x in self.pixels]
        pic2.pix_num = self.pix_num
        pic2.pix_solved = self.pix_solved
        pic2.trc = self.trc[:]
        pic2.tcc = self.tcc[:]
        return pic2


def draw_row(row: List[int]) -> None:
    printing_pattern = { EMPTY: "⬜", FULL: "⬛", UNKNOWN: "ｘ" }
    print(''.join(draw_iter(row, printing_pattern)), end='')


def draw_iter(row, patt):
    yield from (patt[el] for el in row)


def draw(nonogram: Nonogram) -> None:
    print()
    i = 0
    x = ["０","１","２","３","４","５","６","７","８","９"]
    { print(x[a // 10], end='') for a in range(len(nonogram[0])) }
    print()
    { print(x[a % 10], end='') for a in range(len(nonogram[0])) }
    print()
    for row in nonogram:
        draw_row(row)
        print(i)
        i += 1


def short_gen_line(clues: List[int]) -> List[int]:
    result = []
    for clue in clues:
        result += [FULL] * clue + [EMPTY]
    return result[:-1]


def load_picture(filename: str) -> Picture:
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    result = [[FULL if ch == '#' else
               EMPTY if ch == '.' else
               UNKNOWN for ch in line] for line in content]
    pic = Picture(len(content), len(content[0]))
    pic.pixels = result
    return pic


def save_picture(pic: Picture, filename: str) -> None:
    with open(filename, mode='w') as f:
        for line in pic.pixels:
            f.write(''.join('.' if p == EMPTY else
                            '#' if p == FULL else '?' for p in line) + '\n')


def load_clues(filename: str) -> Tuple[List[Clue], List[Clue]]:
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    cols: List[Clue] = list()
    rows: List[Clue] = list()
    for line in content:
        if not line:
            cols.append([])
        elif line[0] == '#':
            pass
        elif line == "---":
            rows, cols = cols, rows
        else:
            cols.append([int(x) for x in line.split()])
    return rows, cols


def clues_valid(rows: List[Clue], cols: List[Clue]) -> bool:
    if sum(map(sum, rows)) != sum(map(sum, cols)):
        print(sum(map(sum, rows)), sum(map(sum, cols)))
        return False
    for row in rows:
        if not check_line([], row, len(cols), False):
            print("reason 2")
            return False
    for col in cols:
        if not check_line([], col, len(rows), False):
            print("reason 3")
            return False
    return True


def gen_line_clues(line: List[int]) -> List[int]:
    if not line:
        return []
    result: List[int] = []
    count = 0
    for val in line + [EMPTY]:
        if val == FULL:
            count += 1
        elif val == EMPTY:
            if count > 0:
                result.append(count)
                count = 0
    return result


def len_gen_lines(clue: Clue, size: int) -> int:
    nulls = size - sum(clue) - len(clue) + 1
    if nulls < 0:
        return 0
    return comb(len(clue) + nulls, nulls)


def gen_lines(clue: Clue, size: int) -> List[List[Pixel]]:
    result = gen_lines_real(clue, size, 0, [])
    assert (len(result) == len_gen_lines(clue, size)) \
            or len_gen_lines(clue, size) > 5 * 10 ** 5
    return result


def gen_lines_real(clue: Clue,
               size: int,
               depth: int,
               generated: List[List[Pixel]],
               mini: int = 0,
               filter_: Optional[Tuple[List[Pixel], Set[int], Set[int]]] = None
               ) -> List[List[Pixel]]:
    if filter_ and filter_[0][0] == 5:
        filter_ = None

    if depth == 0:
        if not clue:
            return [[EMPTY] * size]
        current = [FULL] * clue[0]
        generated = [[EMPTY] * i + current for i
                     in range(mini, size - sum(clue) - len(clue) + 2)]
        if filter_:
            generated = filter_gen_lines(filter_, generated)
        if len(clue) > 1:
            return gen_lines_real(clue, size, 1, generated, 0, filter_)
        result = []
        while generated:
            line = generated.pop()
            result.append(line + [EMPTY] * (size - len(line)))
        if not filter_:
            return result
        return filter_gen_lines(filter_, result)

    elif depth == len(clue):
        for line in generated:
            line += [EMPTY] * (size - len(line))
        if not filter_:
            return generated
        return filter_gen_lines(filter_, generated)

    current = clue[depth] * [FULL]
    result = []
    length = 0
    size_needed = size - sum(clue[depth:]) - len(clue[depth:]) + 1
    while generated:
        line = generated.pop()
# TODO fix slowest line (if possible)
        a = [line + [EMPTY] * (i + 1) + current
             for i in range(size_needed - len(line))]
        result += a
        if len(result) > 5 * 10 ** 5:
            return [[5], [5]]
    generated = result
    if filter_:
        generated = filter_gen_lines(filter_, generated)
    return gen_lines_real(clue, size, depth + 1, generated, 0,
                      filter_)


def filter_gen_lines(filter_: Tuple[List[int], Set[int], Set[int]],
                     lines: List[List[int]]) -> List[List[int]]:
    if filter_[0][0] == 5:
        return lines
    a = len(lines)

    result = []
    _, ones, nulls = filter_
    while lines:
        line = lines.pop()
        for i in nulls:
            if i < len(line) and line[i] != EMPTY:
                break
        else:
            for i in ones:
                if i < len(line) and line[i] != FULL:
                    break
            else:
                result.append(line)

    if result:
        min_len = len(min(result, key=len))
        all_2s = all(v == 2 for v in filter_[0][min_len:])
        if all_2s:
            filter_[0][0] = 5
    return result


def gen_filtered_lines(clue: Clue,
                       size: int,
                       prefix_: List[List[Pixel]],
                       filter_: Optional[Tuple[List[Pixel], Set[int], Set[int]]] = None
                       ) -> List[List[Pixel]]:
    prefix = prefix_[:]
    line, ones, zeroes = filter_
    line2 = line[:]
    if filter_[0][0] == 5:
        zeroes = None
    if zeroes:
        first_zero = line.index(EMPTY)
    if zeroes and clue and first_zero < clue[0]:
        line2[:first_zero] = [EMPTY] * first_zero
        prefix2 = line2[:line2.index(UNKNOWN)]
        if not check_line(prefix2, clue, size, True):
            return []
        current_clue2 = gen_line_clues(prefix2)
        if current_clue2 == clue:
            return [prefix2 + [EMPTY] * (size - len(prefix2))]
        result2 = gen_lines_real(clue, size, len(current_clue2),
                      [prefix2[:-1]], len(prefix2), (line2, ones, zeroes))
        return result2

    if not check_line(prefix, clue, size, True):
        return []
    current_clue = gen_line_clues(prefix)
    if current_clue == clue:
        return [prefix + [EMPTY] * (size - len(prefix))]
    result = gen_lines_real(clue, size, len(current_clue),
                      [prefix[:-1]], len(prefix), (line2, ones, zeroes))
    return result


def solve(rows: List[Clue], cols: List[Clue]) -> Optional[Picture]:
    if not clues_valid(rows, cols):
        return None
    pic = Picture(len(rows), len(cols))
    pic.trc = [ len_gen_lines(x, len(cols)) for i, x in enumerate(rows) ]
    pic.tcc = [ len_gen_lines(x, len(rows)) for i, x in enumerate(cols) ]
    mapped_rows = sorted([(i,
                           x) for i, x in enumerate(rows)],
                         key=lambda x: pic.trc[x[0]], reverse=False)
    mapped_cols = sorted([(i,
                           x) for i, x in enumerate(cols)],
                         key=lambda x: pic.tcc[x[0]], reverse=False)

    pic.cols_to_solve = set(range(pic.width))
    pic.rows_to_solve = set(range(pic.height))
    return solve_real(mapped_rows, mapped_cols, pic, 0, [])


def solve_one(clue: List[int], index: int,
               is_col: bool, pic: Picture):
    line = pic.get_column(index) if is_col else pic.pixels[index][:]
    size = pic.height if is_col else pic.width
    check_two: Set[bool] = {UNKNOWN == x for x in line}

    if all(check_two):
        generated = gen_lines(clue, size)
    elif not any(check_two):
        pic.solved_cols.add(index) if is_col else pic.solved_rows.add(index)
        return True, [], 1, index
    else:
        line_rever = list(reversed(line))
        prefix_front, prefix_back = line[:line.index(UNKNOWN)], line_rever[:line_rever.index(UNKNOWN)]
        rever = len(prefix_back) > len(prefix_front)
        if rever:
            line = line_rever
            clue = list(reversed(clue))
        prefix = prefix_back if rever else prefix_front
        generated = gen_filtered_lines(
            clue,
            size,
            prefix,
            (line,
             {i for i, x in enumerate(line) if x == FULL},
             {i for i, x in enumerate(line) if x == EMPTY}))
        if rever:
            for x in generated:
                x.reverse()

    if not generated:
        return False, [], 0, index
    if len(generated) == 2 and generated[0] == generated[1]:
        return True, [], pic.tcc[index] if is_col else pic.trc[index], index
    return True, get_intersection(generated, index, is_col, pic), len(generated), index


def get_intersection(generated: List[List[int]],
                        index: int, is_col: bool,
                        pic: Picture) -> None:
    result = []
    if is_col:
        for j in range(pic.height):
            if pic.pixels[j][index] == UNKNOWN:
                val = intersection_getter(generated, j)
                if val == FULL or val == EMPTY:
                    result.append((j, index, val))
        return result

    for j in range(pic.width):
        if pic.pixels[index][j] == UNKNOWN:
            val = intersection_getter(generated, j)
            if val == FULL or val == EMPTY:
                result.append((index, j, val))
    return result


def write_intersection(lst, pic, is_row):
    for i, j, val in lst:
        if pic.pixels[i][j] == UNKNOWN:
            if is_row:
                pic.cols_to_solve.add(j)
            else:
                pic.rows_to_solve.add(i)
            pic.pixels[i][j] = val


def intersection_getter(generated: List[List[int]], j: int) -> int:
    val = generated[0][j]
    for line in generated:
        if val != line[j]:
            return 2
    return val


def solve_real(mapped_rows: List[Tuple[int, int, List[int]]],
           mapped_cols: List[Tuple[int, int, List[int]]],
           pic: Picture,
           depth: int,
           back_progress: List[str],
           ) -> Optional[Picture]:

    #draw(pic.pixels)
    if back_progress:
        print("backtrack progress:", ' | '.join(every_second(back_progress)))
    if not solve_check(pic, mapped_rows, mapped_cols):
        return None

    for line in pic.pixels:
        if UNKNOWN in line:
            break
    else:
        return pic

    if depth == 1 and not pic.rows_to_solve:
        pic.rows_to_solve = set(range(pic.height))
    if depth == 2 and not pic.cols_to_solve:
        pic.cols_to_solve = set(range(pic.width))
    if (depth % 2 != 0 and not solve_rows_or_cols(mapped_rows, pic, True, depth)) or \
       (depth % 2 == 0 and not solve_rows_or_cols(mapped_cols, pic, False, depth)):
        return None

    mapped_rows = sorted(mapped_rows,
                         key=lambda x: pic.trc[x[0]], reverse=False)
    mapped_cols = sorted(mapped_cols,
                         key=lambda x: pic.tcc[x[0]], reverse=False)
    if not pic.rows_to_solve and not pic.cols_to_solve:
        return solve_backtrack(mapped_rows, mapped_cols, pic, 0, back_progress)
    return solve_real(mapped_rows, mapped_cols, pic, depth + 1, back_progress)


def solve_rows_or_cols(mapped, pic, is_row, depth):
    solve_these = list()
    for i, clue in mapped:
        complexity = pic.trc[i] if is_row else pic.tcc[i]
        if complexity > 2 ** (depth + 1 if is_row else depth):
            break
        if (is_row and i not in pic.rows_to_solve) or (not is_row and i not in pic.cols_to_solve):
            continue
        pic.rows_to_solve.remove(i) if is_row else pic.cols_to_solve.remove(i)
        if (is_row and i in pic.solved_rows) or (not is_row and i in pic.solved_cols):
            continue
        solve_these.append((clue, i, not is_row, pic))
    with Pool(pool_size) as pool:
        res = pool.starmap(solve_one, solve_these)
    bools = [ x[0] for x in res ]
    vals = [ x[1] for x in res ]
    aaaa = [ (x[2], x[3]) for x in res ]
    for ln, idx in aaaa:
        if is_row:
            pic.trc[idx] = ln
        else:
            pic.tcc[idx] = ln
    if not all(bools):
       return False
    for a in vals:
        write_intersection(a, pic, is_row)
    return True


def solve_backtrack(mapped_rows: List[Tuple[int, int, List[int]]],
                    mapped_cols: List[Tuple[int, int, List[int]]],
                    pic: Picture,
                    depth: int,
                    back_progress: List[str]
                    ) -> Optional[Picture]:
    for i, clue in mapped_rows:
        complexity = pic.trc[i]
        if i not in pic.solved_rows:
            if UNKNOWN not in pic.pixels[i]:
                pic.solved_rows.add(i)
                continue
            lowest_row_comp, row, row_clue = complexity, i, clue
            break

    for i, clue in mapped_cols:
        complexity = pic.tcc[i]
        if i not in pic.solved_cols:
            if UNKNOWN not in pic.get_column(i):
                pic.solved_cols.add(i)
                continue
            lowest_col_comp, col, col_clue = complexity, i, clue
            break

    solve_row = lowest_row_comp < lowest_col_comp
    line = pic.pixels[row][:] if solve_row else pic.get_column(col)
    solutions = gen_filtered_lines(row_clue if solve_row else col_clue,
                                           pic.width if solve_row else pic.height,
                                           line[:line.index(UNKNOWN)],
                                           (line,
                                            {i for i, x in enumerate(line) if x == FULL},
                                            {i for i, x in enumerate(line) if x == EMPTY}))
    if not solutions:
        return None
    probability_null = [0] * len(solutions[0])
    for i in range(len(solutions[0])):
        ones = nulls = 0
        for solution in solutions:
            if solution[i] == 0:
                probability_null[i] += 1
    reverse_enthropy = [max(x, len(solutions) - x) - min(x, len(solutions) - x) for x in probability_null]
    highest_ent = reverse_enthropy.index(min(reverse_enthropy))
    new_sols = [0, 1]
    for val in new_sols:
        pic2 = pic.copy_pic()
        if solve_row:
            pic2.pixels[row][highest_ent] = val
            pic2.cols_to_solve = { highest_ent }
            pic2.rows_to_solve = { row }
        else:
            pic2.pixels[highest_ent][col] = val
            pic2.rows_to_solve = { highest_ent }
            pic2.cols_to_solve = { col }
        result = solve_real(mapped_rows,
                            mapped_cols,
                            pic2,
                            1,
                            back_progress \
                            + [f"row {row if solve_row else highest_ent}, col {col if not solve_row else highest_ent }: {val+1}/2 ({round(100*val/2, 2)}%)",
                            val,
                            2]
                 )
        if result:
            return result
    return None
    """for i, solution in enumerate(solutions):
        pic2 = pic.copy_pic()
        if solve_row:
            write_intersection(get_intersection([solution], row, False, pic2), pic2, True)
            pic2.cols_to_solve = {j for j, x in enumerate(line)
                                  if x == UNKNOWN}
        else:
            write_intersection(get_intersection([solution], col, True, pic2), pic2, False)
            pic2.rows_to_solve = {i for i, x in enumerate(line)
                                  if x == UNKNOWN}
        result = solve_real(mapped_rows,
                            mapped_cols,
                            pic2,
                            1,
                            back_progress \
                            + [f"{f'row {row}' if solve_row else f'col {col}'}: {i+1}/{len(solutions)} ({round(100*i/len(solutions), 2)}%)",
                            i,
                            len(solutions)]
                 )
        if result:
            return result
    return None"""


def every_second(lst):
    i = 2
    p = 0
    q = 1
    for i in range(0, len(lst), 3):
        yield lst[i]
    for i in range(2, len(lst), 3):
        p *= lst[i]
        q *= lst[i]
        p += lst[i - 1]
    yield f"{round(100 * p/q, 10)}%"


def solve_check(pic: Picture,
                 mapped_rows: List[Tuple[int, int, List[int]]],
                 mapped_cols: List[Tuple[int, int, List[int]]]) -> bool:
    for i, clue in mapped_rows:
        line = pic.pixels[i][:]
        if not check_line(line, clue, len(line)):
            return False

    for i, clue in mapped_cols:
        line = pic.get_column(i)
        if not check_line(line, clue, len(line)):
            return False

    return True


def check_line(line: List[int],
                clues: Clue,
                size: int,
                modify: bool = False) -> bool:
    zero_count = size - sum(clues)
    if line.count(FULL) > sum(clues):
        return False
    if line.count(EMPTY) > zero_count:
        return False
    if UNKNOWN in line:
        line = line[:line.index(UNKNOWN)]
    generated = gen_line_clues(line)
    if len(generated) == 0:
        size_needed = sum_plus_len_minus_one(clues)
        if len(line) + size_needed > size:
            return False
        if len(line) + size_needed == size:
            if modify:
                line += short_gen_line(clues)
            return True
        return True
    if len(generated) > len(clues) \
       or (line[-1] == EMPTY and generated != clues[:len(generated)]) \
       or generated[:-1] != clues[:len(generated) - 1] \
       or generated[-1] > clues[len(generated) - 1]:
        return False

    if not modify:
        return True

    clues_used = len(generated)
    if generated != clues[:clues_used]:
        line += [FULL] * (clues[clues_used - 1] - generated[-1])
        generated = gen_line_clues(line)
    if generated == clues:
        return True
    if line[-1] == FULL:
        line.append(EMPTY)
    size_needed = sum_plus_len_minus_one(clues[clues_used:])
    if clues_used == len(clues) - 1 \
       and size_needed == size - len(line):
        line += [FULL] * size_needed
    generated = gen_line_clues(line)
    clues_used = len(generated)
    size_needed = sum_plus_len_minus_one(clues[clues_used:])
    if size - len(line) == size_needed:
        line += short_gen_line(clues[clues_used:])
    return True


def sum_plus_len_minus_one(clue: List[int]) -> int:
    return clue[-1] if len(clue) == 1 else \
        sum(clue) + len(clue) - 1


TEST_FILENAME = "_test_nonograms_"


def test_1() -> None:
    image1 = ("..#....\n"
              ".#.#..#\n"
              "#######\n"
              "..##..#\n"
              "...###.\n"
              "..#.#..\n")

    with open(TEST_FILENAME, "w") as file:
        file.write(image1)

    pic1 = load_picture(TEST_FILENAME)
    assert pic1.width == 7
    assert pic1.height == 6
    assert pic1.pixels == [
        [EMPTY, EMPTY, 1, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, 1, EMPTY, 1, EMPTY, EMPTY, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [EMPTY, EMPTY, 1, 1, EMPTY, EMPTY, 1],
        [EMPTY, EMPTY, EMPTY, 1, 1, 1, EMPTY],
        [EMPTY, EMPTY, 1, EMPTY, 1, EMPTY, EMPTY],
    ]

    image2 = '.#?\n'

    with open(TEST_FILENAME, "w") as file:
        file.write(image2)

    pic2 = load_picture(TEST_FILENAME)
    assert pic2.width == 3
    assert pic2.height == 1
    assert pic2.pixels == [[EMPTY, 1, 2]]

    save_picture(pic1, TEST_FILENAME)
    result = ""
    with open(TEST_FILENAME) as file:
        for line in file:
            result += line.rstrip() + '\n'
        assert result == image1

    save_picture(pic2, TEST_FILENAME)
    with open(TEST_FILENAME) as file:
        lines = file.readlines()
        assert len(lines) == 1
        assert lines[0].rstrip() == '.#?'

    clues = ("# This is a file with clues.\n"
             "# Rows come first:\n"
             "1\n"
             "1 1 1\n"
             "# Comments can be anywhere.\n"
             "7\n"
             "2 1\n"
             "3\n"
             "1 1\n"
             "---\n"
             "1\n"
             "2\n"
             "1 2 1\n"
             "4\n"
             "1 2\n"
             "1 1\n"
             "3\n")

    with open(TEST_FILENAME, "w") as file:
        file.write(clues)

    rows, cols = load_clues(TEST_FILENAME)
    assert rows == [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    assert cols == [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]

    with open(TEST_FILENAME, "w") as file:
        file.write("\n---\n\n")

    rows, cols = load_clues(TEST_FILENAME)
    assert rows == [[]]
    assert cols == [[]]


def test_3() -> None:
    rows = [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    cols = [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]
    assert clues_valid(rows, cols)

    rows = [[1], [1, 1, 1], [7], [1, 1], [1], [1, 1]]
    cols = [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1]]

    assert not clues_valid(rows, cols)

    rows = [[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]]
    cols = [[1], [2], [1, 1, 1, 1], [4], [1, 2], [1, 1], [3]]

    assert not clues_valid(rows, cols)

    rows = [[1], [1, 1, 1], [6], [2, 1], [3], [1, 1]]
    cols = [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]]

    assert not clues_valid(rows, cols)

    assert not clues_valid([[1], [1]], [[1, 1]])

    assert clues_valid([[]], [[]])
    assert clues_valid([[1], []], [[], [1]])

    assert clues_valid([[2], [], [2]], [[1, 1], [2]])


def test_4() -> None:
    assert sorted(gen_lines([1, 3, 2], 9)) == sorted([
        [1, EMPTY, 1, 1, 1, EMPTY, 1, 1, EMPTY],
        [1, EMPTY, 1, 1, 1, EMPTY, EMPTY, 1, 1],
        [1, EMPTY, EMPTY, 1, 1, 1, EMPTY, 1, 1],
        [EMPTY, 1, EMPTY, 1, 1, 1, EMPTY, 1, 1]
    ])

    assert sorted(gen_lines([1], 4)) == sorted([
        [1, EMPTY, EMPTY, EMPTY],
        [EMPTY, 1, EMPTY, EMPTY],
        [EMPTY, EMPTY, 1, EMPTY],
        [EMPTY, EMPTY, EMPTY, 1],
    ])

    assert gen_lines([], 10) == [[EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]]

    assert gen_lines([1, 1, 1, 1], 6) == []

    assert gen_lines([1, 1, 1, 1], 7) == [[1, EMPTY, 1, EMPTY, 1, EMPTY, 1]]

    assert gen_lines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 64) \
        == [[FULL, EMPTY, FULL, FULL, EMPTY, FULL, FULL, FULL, EMPTY, FULL, FULL, FULL, FULL, EMPTY, FULL, FULL, FULL, FULL, FULL, EMPTY,
             FULL, FULL, FULL, FULL, FULL, FULL, EMPTY, FULL, FULL, FULL, FULL, FULL, FULL, FULL, EMPTY,
             FULL, FULL, FULL, FULL, FULL, FULL, FULL, FULL, EMPTY, FULL, FULL, FULL, FULL, FULL, FULL, FULL, FULL, FULL, EMPTY,
             FULL, FULL, FULL, FULL, FULL, FULL, FULL, FULL, FULL, FULL]]

    assert len(gen_lines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 67)) == 286


def test_5() -> None:
    assert sorted(gen_filtered_lines([1, 3, 2], 9, [FULL, EMPTY, FULL])) == sorted([
        [FULL, EMPTY, FULL, FULL, FULL, EMPTY, FULL, FULL, EMPTY],
        [FULL, EMPTY, FULL, FULL, FULL, EMPTY, EMPTY, FULL, FULL],
    ])

    assert sorted(gen_filtered_lines([1], 4, [EMPTY])) == sorted([
        [EMPTY, FULL, EMPTY, EMPTY],
        [EMPTY, EMPTY, FULL, EMPTY],
        [EMPTY, EMPTY, EMPTY, FULL],
    ])

    assert gen_filtered_lines([], 10, [EMPTY, EMPTY, EMPTY]) \
        == [[EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY]]

    assert gen_filtered_lines([1, 1, 1, 1], 7, [EMPTY]) == []

    assert gen_filtered_lines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              1000, [FULL, FULL]) == []

    assert len(gen_filtered_lines([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                  100,
                                  [EMPTY for _ in range(32)])) == 1001


def test_6() -> None:
    pic = solve([[1], [1, 1, 1], [7], [2, 1], [3], [1, 1]],
                [[1], [2], [1, 2, 1], [4], [1, 2], [1, 1], [3]])
    assert pic is not None
    assert pic.width == 7
    assert pic.height == 6
    assert pic.pixels == [
        [EMPTY, EMPTY, FULL, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, FULL, EMPTY, FULL, EMPTY, EMPTY, FULL],
        [FULL, FULL, FULL, FULL, FULL, FULL, FULL],
        [EMPTY, EMPTY, FULL, FULL, EMPTY, EMPTY, FULL],
        [EMPTY, EMPTY, EMPTY, FULL, FULL, FULL, EMPTY],
        [EMPTY, EMPTY, FULL, EMPTY, FULL, EMPTY, EMPTY],
    ]

    assert solve([[2], [], [2]], [[1, 1], [2]]) is None

    pic = solve([[2], [], [2]], [[1, 1], [1, 1]])
    assert pic is not None
    assert pic.width == 2
    assert pic.height == 3
    assert pic.pixels == [[FULL, FULL], [EMPTY, EMPTY], [FULL, FULL]]


def test_7(drawing = False) -> None:
    start = time()
    pic = solve([[1, 3, 1, 1, 1], [1, 1, 1, 2, 2, 2], [1, 3, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1]],
                [[5], [], [5], [1, 1, 1], [1, 1, 1], [1, 1],
                 [], [1], [5], [], [1], [5], [], [1], [5]])
    if not pic:
        print(f"Can't solve: IB111")
        return
    if drawing:
        draw(pic.pixels)
    print(f"IB111: {round(time() - start, 5)}")
    assert pic is not None


def test_functions() -> None:
    start = time()
    test_1()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    pic = Picture(1, 1)
    print("ALL FUNCTIONS  ", round(time() - start, 5))
    solve_real([], [], pic, 1, [], True)


def test_easy(drawing = False) -> None:
    solve_folder("demo_nonograms/easy/", drawing)


def test_medium(drawing = False) -> None:
    solve_folder("demo_nonograms/medium/", drawing)


def test_pika(drawing = False) -> None:
    solve_file('demo_nonograms/impossible/pikachu', drawing)


def test_hard(drawing = False) -> None:
    solve_folder("demo_nonograms/hard/")


def solve_folder(loc, drawing = False) -> None:
    start = time()
    pic = Picture(1, 1)

    onlyfiles = [ join(loc, f) for f in listdir(loc) if isfile(join(loc, f)) ]
    for file in onlyfiles:
        solve_file(file, drawing)

    print()
    print(f"All from {loc}: {round(time() - start, 5)}")
    print()


def solve_file(location, drawing = False):
    rows, cols = load_clues(location)
    if not clues_valid(rows, cols):
        print(f"Invalid clues: {location}")
        return
    start = time()
    pic = solve(rows, cols)
    if not pic:
        print(f"Can't solve: {location}")
        return
    if drawing:
        draw(pic.pixels)
    print(f"{location}: {round(time() - start, 5)}")

def test_random():
    a = [[27, 2, 32], [2, 12, 7, 5, 2, 33], [12, 5, 50], [18, 50], [3, 1, 41, 18], [3, 23, 21, 17], [31, 20, 13], [55, 15], [21, 2, 30, 4, 14], [2, 40, 1, 1, 16], [1, 43, 1, 15], [20, 27, 1, 16], [20, 22, 4, 6, 2, 8], [16, 25, 4, 10], [9, 6, 23, 2, 3, 10], [7, 6, 1, 13, 10, 24], [1, 5, 5, 1, 8, 1, 4, 26], [5, 1, 7, 6, 2, 21], [2, 2, 7, 12, 22], [1, 1, 3, 37], [4, 37], [2, 1, 34], [1, 1, 32], [20, 10], [20, 11], [20, 2, 6], [22, 6], [18, 6], [12, 4, 6], [5, 8, 3, 5], [9, 7, 1, 5], [9, 8, 5], [1, 4, 2, 5, 5], [12, 1, 1, 1, 5], [16, 1, 5], [17, 5], [1, 18, 5], [22, 5], [22, 4], [22, 1, 4], [24, 4], [22, 12, 4], [24, 17, 4], [26, 20, 4], [25, 22, 4], [1, 22, 1, 22, 4], [26, 26, 5], [45, 6, 5], [45, 5, 5], [1, 1, 41, 5, 5], [1, 2, 39, 5, 5], [2, 2, 47, 6], [1, 49, 6], [1, 49, 6], [5, 40, 6], [3, 42, 6], [42, 7], [44, 7], [46, 7], [47, 7], [48, 7], [48, 7], [9, 38, 7], [1, 9, 7, 27, 6], [2, 10, 1, 2, 3, 28, 6], [3, 9, 8, 28, 6], [3, 7, 5, 2, 28, 6], [3, 7, 3, 1, 27, 5], [2, 6, 1, 1, 1, 27, 5], [3, 6, 4, 1, 27, 5], [3, 2, 8, 4, 2, 28, 5], [2, 2, 8, 8, 28, 5], [2, 2, 9, 2, 4, 28, 5], [3, 2, 9, 2, 2, 28, 5], [4, 3, 9, 2, 2, 28, 5], [3, 1, 2, 9, 2, 2, 28, 5], [3, 1, 2, 8, 2, 1, 28, 5], [2, 2, 2, 8, 2, 1, 28, 5], [2, 2, 3, 9, 3, 2, 28, 5], [3, 2, 4, 9, 2, 2, 27, 5], [3, 2, 4, 12, 2, 2, 26, 6], [6, 6, 12, 2, 2, 26, 5], [7, 4, 2, 13, 1, 1, 26, 5], [9, 3, 3, 13, 1, 2, 26, 5], [6, 2, 2, 4, 2, 10, 2, 25, 5], [4, 2, 3, 4, 5, 1, 7, 2, 25, 4], [2, 3, 3, 2, 3, 8, 4, 3, 1, 5, 1, 24, 4], [3, 2, 1, 4, 8, 4, 3, 2, 24, 4], [3, 2, 5, 8, 4, 5, 2, 24, 4], [3, 2, 5, 8, 3, 3, 1, 23, 4], [3, 3, 3, 14, 1, 1, 23, 4], [3, 1, 6, 15, 1, 23, 4], [3, 2, 6, 12, 1, 22, 4], [2, 2, 4, 1, 12, 1, 22, 4], [1, 2, 6, 12, 2, 22, 4], [2, 5, 7, 12, 2, 21, 4], [3, 7, 6, 12, 2, 6, 13, 4], [3, 7, 5, 12, 1, 4, 13, 4], [5, 6, 5, 12, 1, 13, 4], [3, 2, 2, 1, 8, 4, 13, 5], [2, 3, 4, 2, 1, 9, 13, 6], [2, 6, 3, 2, 10, 13, 6], [8, 3, 10, 13, 6], [8, 3, 11, 3, 13, 6], [10, 5, 12, 3, 7, 13, 2, 6], [10, 7, 12, 3, 2, 3, 2, 13, 2, 6], [5, 4, 7, 13, 2, 2, 2, 13, 3, 6], [3, 2, 1, 4, 2, 7, 14, 2, 3, 3, 12, 9], [6, 2, 6, 9, 15, 2, 1, 2, 2, 12, 1, 1, 7], [6, 2, 5, 11, 15, 2, 2, 2, 2, 12, 2, 1, 6], [3, 2, 3, 4, 14, 16, 1, 2, 3, 2, 3, 12, 3, 6], [3, 9, 4, 16, 17, 1, 3, 2, 2, 3, 12, 2, 6], [7, 6, 4, 18, 17, 3, 3, 4, 2, 3, 12, 2, 1, 6], [7, 2, 3, 2, 20, 16, 2, 3, 4, 2, 15, 11, 5, 1, 6], [7, 2, 3, 21, 17, 2, 3, 4, 23, 11, 1, 2, 1, 2, 6], [5, 3, 23, 17, 2, 1, 4, 28, 11, 2, 1, 2, 6], [3, 4, 23, 16, 1, 2, 35, 11, 1, 4, 7], [1, 7, 6, 23, 17, 2, 4, 37, 10, 2, 3, 7], [1, 9, 3, 22, 17, 1, 2, 1, 39, 10, 1, 1, 3, 6], [1, 10, 2, 24, 18, 2, 1, 2, 38, 11, 1, 1, 3, 6], [1, 10, 27, 19, 2, 1, 2, 3, 31, 10, 1, 5, 6], [1, 2, 7, 27, 19, 2, 1, 2, 3, 31, 10, 1, 5, 6], [1, 2, 7, 27, 19, 2, 2, 2, 3, 31, 10, 2, 1, 3, 1, 6], [1, 1, 5, 30, 19, 2, 1, 1, 3, 30, 10, 1, 1, 1, 2, 6], [2, 2, 1, 30, 22, 2, 1, 1, 30, 11, 1, 1, 1, 1, 1, 6], [2, 2, 1, 30, 23, 2, 1, 1, 2, 29, 11, 3, 2, 2, 9], [2, 2, 1, 29, 2, 23, 1, 2, 1, 3, 2, 28, 13, 2, 2, 1, 2, 9], [3, 4, 29, 2, 25, 1, 2, 2, 2, 34, 14, 2, 1, 14], [3, 32, 27, 2, 3, 2, 2, 2, 4, 27, 19, 3, 14], [4, 32, 29, 3, 2, 1, 3, 1, 3, 66], [37, 32, 2, 1, 3, 2, 3, 66], [7, 27, 35, 4, 3, 1, 2, 1, 3, 66], [35, 36, 4, 3, 4, 2, 4, 1, 65], [33, 2, 47, 4, 2, 6, 2, 1, 65], [33, 54, 2, 3, 6, 63], [33, 55, 5, 4, 73], [175], [175], [175], [175]], [[22], [1, 2, 16], [2, 2, 2, 4, 1, 4, 13], [3, 4, 9, 5, 11], [3, 6, 15, 10], [3, 2, 2, 3, 6, 4, 17, 17], [22, 22, 2, 27], [30, 10, 7, 6, 4, 8], [3, 8, 2, 1, 5, 3, 22], [3, 1, 4, 5, 1, 5, 12], [9, 5, 15, 15, 14], [63], [4, 13, 3, 3, 18], [1, 4, 8, 3, 4, 17], [3, 1, 3, 3, 3, 18], [3, 2, 19], [19], [19], [20], [1, 1, 1, 20], [1, 1, 3, 2, 3, 21], [4, 3, 4, 5, 5, 1, 22], [2, 3, 8, 8, 35], [70], [3, 7, 15, 6, 5, 24], [1, 7, 5, 6, 3, 25], [3, 3, 1, 4, 26], [2, 2, 2, 26], [26], [27], [27], [27], [28], [21, 4], [22, 5], [20, 7], [20, 7], [18, 6], [18, 8], [17, 11], [17, 11], [14, 14], [11, 14], [11, 12], [10, 13], [8, 13], [6, 20], [1, 5, 22], [1, 25], [2, 28], [2, 32], [3, 33], [4, 35], [5, 36], [5, 38], [2, 5, 39], [3, 5, 42], [3, 4, 44], [3, 4, 45], [4, 51], [4, 50], [55], [53], [52], [10, 8, 19], [10, 3, 16], [10, 18], [8, 20], [7, 4, 3, 10], [5, 4, 11], [3, 4, 12], [2, 5, 9], [6, 2, 9], [5, 3, 10], [2, 4, 2, 7], [2, 2, 2, 8], [1, 2, 1, 8], [2, 2, 1, 9], [3, 1, 2, 9], [2, 1, 7], [4, 1, 7], [3, 7], [5, 12], [2, 2, 9], [2, 1, 9], [3, 6], [3, 9], [5, 1, 8], [6, 3, 11], [6, 13, 4], [7, 5, 6], [6, 7], [8, 2, 5], [2, 8, 3, 5], [1, 7, 1, 4, 5], [1, 1, 6, 1, 4, 4], [9, 6, 1, 4], [5, 7, 3, 7, 1, 5], [14, 8, 1, 7], [1, 2, 12, 2, 8, 5, 10], [2, 2, 13, 1, 8, 6, 1, 12], [3, 6, 15, 7, 16, 4], [1, 3, 2, 5, 17, 8, 3, 5, 2, 5], [2, 12, 20, 8, 5, 2, 6], [1, 1, 4, 8, 21, 7, 4, 1, 8], [4, 4, 8, 22, 7, 14, 2, 7], [9, 5, 22, 1, 8, 15, 2, 6], [15, 24, 10, 14, 1, 7], [4, 8, 24, 1, 10, 15, 5], [4, 9, 1, 23, 13, 21, 5], [18, 21, 11, 23, 5], [4, 12, 21, 11, 29], [17, 21, 12, 4, 1, 27], [17, 32, 4, 24], [17, 31, 2, 32], [2, 12, 31, 2, 33], [1, 11, 29, 3, 3, 25], [1, 12, 9, 16, 4, 2, 25], [1, 11, 6, 15, 4, 4, 10, 2, 26], [1, 6, 4, 2, 3, 17, 8, 6, 6, 2, 26], [13, 22, 3, 8, 2, 2, 26], [11, 36, 1, 2, 26], [2, 4, 2, 2, 18, 2, 7, 1, 2, 26], [2, 7, 1, 22, 2, 1, 4, 26], [12, 2, 24, 3, 33], [17, 2, 20, 8, 32], [16, 2, 2, 20, 29], [1, 14, 6, 21, 26], [1, 19, 21, 26], [19, 22, 19, 26], [18, 51, 26], [6, 12, 54, 26], [5, 12, 56, 26], [16, 57, 25], [15, 57, 12, 11], [15, 57, 6, 12], [14, 57, 3, 13], [16, 3, 2, 56, 14], [14, 3, 2, 55, 16], [14, 4, 1, 2, 55, 20], [21, 2, 3, 6, 90], [21, 11, 6, 89], [33, 7, 89], [32, 98], [16, 15, 96], [14, 1, 11, 2, 94], [9, 6, 12, 93], [9, 2, 3, 18, 91], [10, 1, 1, 18, 89], [9, 2, 1, 15, 69, 14], [9, 2, 2, 14, 66, 12], [9, 1, 2, 15, 61, 1, 12], [10, 2, 2, 14, 54, 14], [4, 1, 1, 2, 1, 13, 3, 31, 13], [4, 1, 16, 1, 27, 2, 11], [4, 1, 16, 1, 27, 2, 11], [4, 1, 17, 21, 1, 4, 2, 12], [5, 1, 1, 17, 18, 2, 18], [6, 1, 1, 15, 3, 2, 12], [6, 1, 2, 15, 2, 11], [6, 1, 4, 12, 3, 4, 11], [6, 5, 10, 1, 4, 15], [13, 8, 5, 13], [13, 8, 1, 2, 3, 2, 15], [12, 8, 2, 3, 2, 23], [12, 12, 3, 2, 5, 13], [12, 12, 17, 15], [25, 1, 2, 17], [25, 7, 2, 2, 15], [29, 16, 39], [38, 39, 41], [140], [140], [140], [140]]
    if not clues_valid(a[0], a[1]):
        print(f"Invalid clues")
        return
    start = time()
    pic = solve(a[0], a[1])
    if not pic:
        print("Can't solve")
        return
    draw(pic.pixels)
    print(f"big: {round(time() - start, 5)}")
