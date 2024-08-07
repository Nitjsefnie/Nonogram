from new import *
from solver import *
from os import system
from pool import reset_pool

def foo():
    i = 16900
    not_found = 0
    found = 0
    draw_steps(False)
    while not_found < 100 and found < 500:
        i += 1
        text = fetch_webpbn(i)
        if text is not None:
            found += 1
            not_found = 0
            print(i, "found, solving")
            with open(f"demo_nonograms/webpbn/{i}", "w") as file:
                file.write(text)
            reset_pool()
            start = time()
            solve_file(f"demo_nonograms/webpbn/{i}", drawer = None)
            solve_time = time() - start
            folder = "easiest" if solve_time < 0.1 else "easy" if solve_time < 5 else "medium" if solve_time < 30 else "hard" if solve_time < 600 else "extreme" if solve_time < 21600 else "insane"
            with open(f"demo_nonograms/{folder}/{i}", "w") as file:
                file.write(text)
            print(time() - start)
            system(f"rm demo_nonograms/webpbn/{i}")
        else:
            print(i, "not found")
            not_found += 1


foo()
terminate_pool()
