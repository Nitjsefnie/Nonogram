from multiprocessing import Pool
from os import cpu_count

from globalvars import Global

cpdef void change_pool_size(int size = cpu_count() - 1):
    terminate_pool()
    Global.pool_size = min(size, cpu_count() - 1)
    Global.pool = Pool(Global.pool_size)

cpdef void terminate_pool():
    if Global.pool is not None:
        Global.pool.terminate()
        Global.pool.join()
        Global.pool = None

cpdef void reset_pool():
    change_pool_size(Global.pool_size)