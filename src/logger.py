import sys
import time
import warnings

warnings.filterwarnings("ignore")
from functools import wraps
import logging

notebook = False

if notebook:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        stream=sys.stdout)
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def logit(func):  # print message when the function is called
    @wraps(func)
    def with_logging(*args, **kwargs):  # to remember *args call list unpacking *kwargs call dict unpacking
        logging.info(
            '{} : {}, {}, ({})'.format(func.__name__, func.__doc__, args, kwargs))  # if name and doc exit else None
        return func(*args, **kwargs)

    return with_logging


def timeit(func):  # a decorator to record the function running time
    @wraps(func)
    def wrap(*args, **kwargs):
        t0 = time.time()  # init time
        res = func(*args, **kwargs)
        logging.info('{}() time: ({}s)'.format(func.__name__, time.time() - t0))
        return res

    return wrap
