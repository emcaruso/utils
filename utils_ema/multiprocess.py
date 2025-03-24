from typing import Callable, Iterable, Optional, Any
import multiprocessing as mp
import sys

# NOTE(alberto): allow usage without tqdm installed
try:
    import tqdm
except ImportError:
    pass


def __safe_wrapper(fn: Callable) -> Callable:
    def wrapper(args: tuple) -> Any:
        try:
            return fn(*args)
        except Exception as e:
            return e

    return wrapper


def __unpack(input: tuple[Callable, tuple]):
    func, args = input
    safe_func = __safe_wrapper(func)
    return safe_func(args)


# iterator wrapper used to propagate exceptions to the main process
class _IteratorWrapper:
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.it)
        if isinstance(item, Exception):
            raise item
        return item


def map_unordered(fn: Callable, input: Iterable, jobs: Optional[int] = None) -> list:
    """Calls fn on each element of input in parallel (using multiprocessing module) using 'jobs' processes

    Args:
        fn (Callable): function to be executed in parallel
        input (Iterable): iterable yielding arguments to fn as tuples
        jobs (int): number of processes to use. None means use all available cores

    Returns:
        list: list of results of calling fn on each element of input. NOTE: the order of the results is not guaranteed to match the order of the input.
    """
    data = [(fn, d) for d in input]
    with mp.Pool(jobs) as pool:
        iter = pool.imap_unordered(__unpack, data)

        # NOTE(alberto): show progress bar if tqdm is available
        if "tqdm" in sys.modules:
            iter = tqdm.tqdm(iter, total=len(data))
            iter = iter.iterable

        exception_filter = _IteratorWrapper(iter)
        results = list(exception_filter)
    return results

def run_function_in_parallel(fn: Callable, inputs: Iterable, args: Iterable = None):
    processes = []

    for input in inputs:
        p = mp.Process(target=fn, args=(input,*args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def run_in_multiprocess(func):
    def wrapper(*args, **kwargs):
        # Create a process to run the function
        process = mp.Process(target=func, args=args, kwargs=kwargs)
        process.start()
        # process.join()  # Wait for the process to finish
    return wrapper
