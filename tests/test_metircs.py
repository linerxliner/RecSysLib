#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pytest

from _exceptions import DimensionError, EmptyArrayError
from metrics import rmse, mae


class TestMetrics(object):
    def test_rmse(self):
        assert math.sqrt(5) / 2 == rmse(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))
        with pytest.raises(EmptyArrayError):
            rmse(np.array([]), np.array([]))
        with pytest.raises(EmptyArrayError):
            rmse(np.array([]), np.array([1]))
        with pytest.raises(DimensionError):
            rmse(np.array([[1], [2]]), np.array([1, 2]))
        with pytest.raises(ValueError):
            rmse(np.array([1, 2]), np.array([3, 4, 5]))

    def test_mae(self):
        assert 2 == mae(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))
        with pytest.raises(EmptyArrayError):
            mae(np.array([]), np.array([]))
        with pytest.raises(EmptyArrayError):
            mae(np.array([]), np.array([1]))
        with pytest.raises(DimensionError):
            mae(np.array([[1], [2]]), np.array([1, 2]))
        with pytest.raises(ValueError):
            mae(np.array([1, 2]), np.array([3, 4, 5]))
