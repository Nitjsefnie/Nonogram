import os
from datetime import datetime
from multiprocessing import freeze_support
from os import getpid
from random import choice

from pool import terminate_pool, change_pool_size
from tests import *
from utils import gen_line_clues

SIZE = 30
# PIXELS = [0, 0, 1, 1, 1]
PIXELS = [0, 1]
THREAD_COUNT = 12


def generate_nonogram(height, width):
    pic = Picture(height, width)
    pic.set_all_pixels(array([
        [choice(PIXELS) for _ in range(width)] for _ in range(height)
    ]))
    return pic


def generate_clues(pic):
    rows, cols = list(), list()
    for row in pic.get_pixels():
        rows.append(gen_line_clues(row))
    for i in range(pic.width):
        cols.append(gen_line_clues(pic.get_col(i)))
    return rows, cols

if __name__ == '__main__':
    freeze_support()
    change_pool_size(THREAD_COUNT)
    while True:
        pic = generate_nonogram(SIZE, SIZE)
        row_clues, col_clues = generate_clues(pic)
        os.makedirs('random_outputs', exist_ok=True)
        save_clues("random_outputs/current." + str(getpid()), row_clues, col_clues)
        print(datetime.now().strftime("%H:%M:%S"), end="\r")
        start = time()
        pic2 = list(solve(row_clues, col_clues))
        result_time = time() - start
        print(getpid(), result_time, len(pic2))
        if result_time < 300:
            continue
        os.makedirs(f'random_outputs/{str(SIZE)}', exist_ok=True)
        save_clues(
            "random_outputs/" +
            str(SIZE) +
            "/" +
            str(result_time),
            row_clues,
            col_clues)
    terminate_pool()
