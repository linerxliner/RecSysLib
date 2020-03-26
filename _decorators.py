#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import numpy as np

from _exceptions import UnexpectedDimensionError, EmptyArrayError, DecoratorArgumentError


def expected_arrays(non_empty=None, dim=None):
    """Decorator to force numpy arrays to be non-empty or specified dim."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if non_empty or dim:
                arrays = []

                for arg in args:
                    if isinstance(arg, np.ndarray):
                        arrays.append(arg)
                for arg in kwargs.values():
                    if isinstance(arg, np.ndarray):
                        arrays.append(arg)

                for i in range(min(len(non_empty or dim), len(arrays))):
                    if non_empty and non_empty[i] and arrays[i].size == 0:
                        raise EmptyArrayError
                    if dim and dim[i] > 0 and dim[i] != arrays[i].ndim:
                        raise UnexpectedDimensionError(dim[i], arrays[i].ndim)

            return func(*args, **kwargs)
        return wrapper

    if non_empty is not None and not isinstance(non_empty, (tuple, list)):
        non_empty = [non_empty]
    if dim is not None and not isinstance(dim, (tuple, list)):
        dim = [dim]

    if non_empty is not None and callable(non_empty[0]) and dim is None:
        actual_func = non_empty[0]
        non_empty = None
        return decorator(actual_func)
    else:
        if non_empty and dim and len(non_empty) != len(dim):
            raise DecoratorArgumentError('"non_empty" and "dim" have different length.')
        if non_empty:
            for arg in non_empty:
                if not isinstance(arg, bool):
                    raise DecoratorArgumentError('Args in non_empty must be bool.')
        if dim:
            for arg in dim:
                if not isinstance(arg, int):
                    raise DecoratorArgumentError('Args in dim must be int.')
        return decorator
