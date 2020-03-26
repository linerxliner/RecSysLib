#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from _decorators import expected_arrays


@expected_arrays(non_empty=[True, True], dim=[1, 1])
def rmse(result, target):
    """Root mean squared error."""
    return np.sqrt(np.sum((result - target) ** 2)) / result.size


@expected_arrays(non_empty=[True, True], dim=[1, 1])
def mae(result, target):
    """Mean absolute error."""
    return np.sum(np.abs(result - target)) / result.size
