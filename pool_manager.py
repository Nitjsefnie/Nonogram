from multiprocessing import Pool
from os import cpu_count

class PoolManager:
    """Manage a multiprocessing Pool with configurable size."""

    def __init__(self):
        self.pool_size = 1
        self.pool = None

    def change_pool_size(self, size=cpu_count() - 1):
        self.terminate_pool()
        self.pool_size = min(max(size, 1), max(cpu_count() - 1, 1))
        self.pool = Pool(self.pool_size)

    def terminate_pool(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None

    def reset_pool(self):
        self.change_pool_size(self.pool_size)

pool_manager = PoolManager()
