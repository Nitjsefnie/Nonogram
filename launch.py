from tests import *
from new import *

change_pool_size()
draw_steps(True)

if __name__ == "__main__":
    #2647  2712  3867  3929
    solve_folder("demo_nonograms/webpbn")
    pass

terminate_pool()

