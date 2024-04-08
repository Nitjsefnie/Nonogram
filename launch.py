#!/bin/python3
from tests import *
from new import *

change_pool_size(12)
#draw_steps(True)

if __name__ == "__main__":
    files = [
        #"15/313.80157017707825",
    ]
    for file in files:
        solve_file("random_outputs/" + file)
    solve_folder("random_outputs/15")
    pass

terminate_pool()

