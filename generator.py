SIZE = 15
#PIXELS = [0, 0, 1, 1, 1]
PIXELS = [0, 1]
THREAD_COUNT = 12

from tests import *
from random import choice
from os import getpid
from datetime import datetime

#draw_steps(True)

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

change_pool_size(THREAD_COUNT)
while True:
    pic = generate_nonogram(SIZE, SIZE)
    row_clues, col_clues = generate_clues(pic)
    save_clues("random_outputs/current." + str(getpid()), row_clues, col_clues)
    print(datetime.now().strftime("%H:%M:%S"), end="\r")
    start = time()
    pic2 = list(solve(row_clues, col_clues))
    result_time = time() - start
    print(getpid(), result_time, len(pic2))
    if result_time < 300:
        continue
    #if pic.pixels != pic2.pixels:
        #print("pics differ")
        #draw(pic.pixels)
        #draw(pic2.pixels)
    save_clues("random_outputs/" + str(SIZE) + "/" + str(result_time), row_clues, col_clues)
terminate_pool()
