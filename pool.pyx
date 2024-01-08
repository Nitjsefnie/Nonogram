from globalvars import Global
from os import cpu_count
from multiprocess import Pool

cpdef void change_pool_size(int size = cpu_count() - 1):
    if Global.pool is not None:
        terminate_pool()
    Global.pool_size = min(size, cpu_count() - 1)
    Global.pool = Pool(Global.pool_size)

cpdef void terminate_pool():
    if Global.pool is not None:
        Global.pool.terminate()
        Global.pool = None
