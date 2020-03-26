#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import pytest

from metrics import *


class TestRMSE(object):
    def test_calc(self):
        assert math.sqrt(5) / 2 == rmse(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))

    def test_diff_lens(self):
        with pytest.raises(ValueError):
            rmse(np.array([1, 2]), np.array([3, 4, 5]))


class TestMAE(object):
    def test_calc(self):
        assert 2 == mae(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1]))

    def test_diff_lens(self):
        with pytest.raises(ValueError):
            mae(np.array([1, 2]), np.array([3, 4, 5]))
