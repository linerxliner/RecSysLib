#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from _exceptions import DimensionError, EmptyArrayError

import numpy as np


def rmse(result, target):
    """Root mean squared error."""
    if result.size == 0 or target.size == 0:
        raise EmptyArrayError()
    if result.ndim != 1 or target.ndim != 1:
        raise DimensionError(1)

    return np.sqrt(np.sum((result - target) ** 2)) / result.size


def mae(result, target):
    """Mean absolute error."""
    if result.size == 0 or target.size == 0:
        raise EmptyArrayError()
    if result.ndim != 1 or target.ndim != 1:
        raise DimensionError(1)

    return np.sum(np.abs(result - target)) / result.size
