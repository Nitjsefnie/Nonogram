from multiprocess import freeze_support, Pool


def worker_function(x):
    # Your worker function code here
    return x * x


if __name__ == '__main__':
    freeze_support()

    # Your main code here
    with Pool() as pool:
        results = pool.map(worker_function, [1, 2, 3, 4, 5])
        print(results)
