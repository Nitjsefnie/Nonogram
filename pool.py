from multiprocessing import Pool
from os import cpu_count

from globalvars import Global


def change_pool_size(size=cpu_count() - 1):
    terminate_pool()
    max_workers = max(cpu_count() - 1, 1)
    Global.pool_size = min(max(size, 1), max_workers)
    Global.pool = Pool(Global.pool_size)


def terminate_pool():
    if Global.pool is not None:
        Global.pool.terminate()
        Global.pool.join()
        Global.pool = None


def reset_pool():
    change_pool_size(Global.pool_size)