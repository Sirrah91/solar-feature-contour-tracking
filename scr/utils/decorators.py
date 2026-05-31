import numpy as np
from functools import wraps
import time
import traceback
from typing import Callable


def timing(func=None, num_repeats: int = 1):
    """
    Define parametrized decorator with arguments
    """

    def _timestamp(t: float, prec: int = 3) -> str:
        if t < 0.:
            raise ValueError('Elapsed time "t" must be a non-negative number.')
        n = 3.

        if t * n < 1.:  # less than 1/n seconds -> show milliseconds
            return f"{np.round(t * 1000., prec):.{prec:d}f} milliseconds"
        if t / 60. < n:  # less than n minutes -> show seconds
            return f"{np.round(t, prec):.{prec:d}f} seconds"
        if t / 60. < n * 60.:  # between n minutes and n hours -> show minutes
            return f"{np.round(t / 60., prec):.{prec:d}f} minutes"
        if t / 3600. < n * 24.:  # between n hours and n days -> show hours
            return f"{np.round(t / 3600., prec):.{prec:d}f} hours"
        return f"{np.round(t / 86400., prec):.{prec:d}f} days"  # show days

    if not (callable(func) or func is None):
        raise ValueError('The usage of timing decorator is "@timing", "@timing()", "@timing(num_repeats=integer)", '
                         '"@timing(func=None, num_repeats=integer)", '
                         'or timing(func, num_repeats=integer)(*args, **kwargs)')

    if num_repeats < 1 or not isinstance(num_repeats, int):
        raise ValueError(f'"num_repeats" must be positive integer but is {num_repeats}')

    def _decorator(f: Callable):
        """
        Repeats execution "num_repeats" times and measures elapsed time
        """

        @wraps(f)
        def wrap(*args, **kw):
            ts = time.time()
            for _ in range(num_repeats - 1):
                f(*args, **kw)
            else:
                result = f(*args, **kw)
            te = time.time()

            elapsed_time = _timestamp(te - ts, prec=3)
            if num_repeats == 1:
                print(f'Function "{f.__name__}" took {elapsed_time}.')
            else:
                elapsed_time_per_repetition = _timestamp((te - ts) / num_repeats, prec=5)
                print(f'Function "{f.__name__}" took {elapsed_time} after {num_repeats} repetitions '
                      f'({elapsed_time_per_repetition} per repetition).')

            return result

        return wrap

    return _decorator(func) if callable(func) else _decorator


def reduce_like(func: Callable):
    @wraps(func)
    def _decorator(*args, **kw):
        args = list(args)
        arrays = args[0]
        result = arrays[0]

        for array in arrays[1:-1]:
            if args[1:]:
                new_args = [(result, array), *args[1:]]
            else:
                new_args = [(result, array)]
            result = func(*new_args, **kw)
        else:
            if args[1:]:
                new_args = [(result, arrays[-1]), *args[1:]]
            else:
                new_args = [(result, arrays[-1])]

            return func(*new_args, **kw)

    decorated_function = func
    decorated_function.reduce = _decorator

    return decorated_function


def safe_call(func=None):
    def _decorator(f: Callable):

        @wraps(f)
        def wrap(*args, **kw):
            try:
                return f(*args, **kw)

            except Exception:
                print(traceback.format_exc())

        return wrap

    return _decorator(func) if callable(func) else _decorator
