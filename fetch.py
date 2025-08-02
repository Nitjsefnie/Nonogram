import os
from multiprocessing import freeze_support

from new import *
from solver import *
from pool import reset_pool, terminate_pool, change_pool_size


def foo():
    change_pool_size()
    i = 1380
    not_found = 0
    found = 0
    max_sols = 2000
    while not_found < 100 and found < 500:
        i += 1
        text = fetch_webpbn(i)
        if text is not None:
            found += 1
            not_found = 0
            print(i, "found, solving...", end=' ')
            os.makedirs('demo_nonograms/webpbn', exist_ok=True)
            with open(f"demo_nonograms/webpbn/{i}", "w") as file:
                file.write(text)
            reset_pool()
            start = time()
            solutions = solve_file(f"demo_nonograms/webpbn/{i}", number=max_sols)
            solve_time = time() - start
            if solutions == max_sols:
                folder = "unsolved"
            else:
                folder = "easy" if solve_time < 5 else "medium" if solve_time < 30 else "hard" if solve_time < 600 \
                    else "extreme" if solve_time < 21600 else "insane"
            subfolder = "single" if solutions == 1 else "multi"
            os.makedirs(f'demo_nonograms/{folder}/{subfolder}', exist_ok=True)
            with open(f"demo_nonograms/{folder}/{subfolder}/{i}", "w") as file:
                file.write(text)
            # print(time() - start)
            os.remove(f"demo_nonograms/webpbn/{i}")
        else:
            print(i, "not found")
            not_found += 1


if __name__ == '__main__':
    freeze_support()
    foo()
    terminate_pool()
