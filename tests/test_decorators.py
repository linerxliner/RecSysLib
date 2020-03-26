#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from _decorators import *
from _exceptions import DecoratorArgumentError


class TestExpectedArrays(object):
    @pytest.mark.parametrize('ea', [expected_arrays, expected_arrays()])
    def test_default(self, ea):
        @ea
        def sum1(a):
            return np.sum(a)
        assert 3 == sum1(np.array([1, 2]))
        assert 0 == sum1(np.array([]))
        assert 10 == sum1(np.array([[1, 2], [3, 4]]))

        @ea
        def sum2(a, b):
            return np.sum(a + b)
        assert 10 == sum2(np.array([1, 2]), np.array([3, 4]))
        with pytest.raises(ValueError):
            sum2(np.array([]), np.array([3, 4]))
        assert 36 == sum2(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

    def test_one_array_non_empty(self):
        @expected_arrays(non_empty=True)
        def sum1(a):
            return np.sum(a)
        assert 3 == sum1(np.array([1, 2]))
        with pytest.raises(EmptyArrayError):
            sum1(np.array([]))
        assert 10 == sum1(np.array([[1, 2], [3, 4]]))

    @pytest.mark.parametrize('non_empty', [[True, True], [False, True], [True, False], [False, False]])
    def test_two_arrays_non_empty(self, non_empty):
        @expected_arrays(non_empty=non_empty)
        def sum1(a, b):
            return np.sum(a + b)
        assert 10 == sum1(np.array([1, 2]), np.array([3, 4]))
        with pytest.raises(EmptyArrayError if non_empty[0] else ValueError):
            sum1(np.array([]), np.array([3, 4]))
        with pytest.raises(EmptyArrayError if non_empty[1] else ValueError):
            sum1(np.array([1, 2]), np.array([]))
        if non_empty != [False, False]:
            with pytest.raises(EmptyArrayError):
                sum1(np.array([]), np.array([]))
        assert 36 == sum1(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))

    def test_one_array_dim(self):
        @expected_arrays(dim=1)
        def sum1(a):
            return np.sum(a)
        assert 3 == sum1(np.array([1, 2]))
        with pytest.raises(UnexpectedDimensionError):
            sum1(np.array([[]]))
        with pytest.raises(UnexpectedDimensionError):
            sum1(np.array([[1, 2], [3, 4]]))
        assert 0 == sum1(np.array([]))

        @expected_arrays(dim=2)
        def sum2(a):
            return np.sum(a)
        assert 10 == sum2(np.array([[1, 2], [3, 4]]))
        with pytest.raises(UnexpectedDimensionError):
            sum2(np.array([]))
        with pytest.raises(UnexpectedDimensionError):
            sum2(np.array([1, 2]))
        assert 0 == sum2(np.array([[]]))

    @pytest.mark.parametrize('dim', [[1, 1], [1, 2], [2, 1], [2, 2]])
    def test_two_arrays_dim(self, dim):
        @expected_arrays(dim=dim)
        def sum1(a, b):
            return np.sum(a + b)
        if dim == [1, 1]:
            assert 10 == sum1(np.array([1, 2]), np.array([3, 4]))
        else:
            with pytest.raises(UnexpectedDimensionError):
                sum1(np.array([1, 2]), np.array([3, 4]))
        if dim == [1, 2]:
            assert 24 == sum1(np.array([1, 2]), np.array([[3, 4], [5, 6]]))
        else:
            with pytest.raises(UnexpectedDimensionError):
                sum1(np.array([1, 2]), np.array([[3, 4], [5, 6]]))
        if dim == [2, 1]:
            assert 32 == sum1(np.array([[1, 2], [3, 4]]), np.array([5, 6]))
        else:
            with pytest.raises(UnexpectedDimensionError):
                sum1(np.array([[1, 2], [3, 4]]), np.array([5, 6]))
        if dim == [2, 2]:
            assert 36 == sum1(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        else:
            with pytest.raises(UnexpectedDimensionError):
                sum1(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        assert 0 == sum1(np.array([] if dim[0] == 1 else [[]]), np.array([] if dim[1] == 1 else [[]]))

    def test_diff_arg_list_len(self):
        with pytest.raises(DecoratorArgumentError):
            @expected_arrays(non_empty=True, dim=[1, 2])
            def sum1(a, b):
                return np.sum(a + b)

        with pytest.raises(DecoratorArgumentError):
            @expected_arrays(non_empty=[False, True], dim=1)
            def sum2(a, b):
                return np.sum(a + b)

    def test_two_arrays_non_empty_and_dim(self):
        @expected_arrays(non_empty=[True, True], dim=[1, 1])
        def sum1(a, b):
            return np.sum(a + b)
        assert 21 == sum1(np.array([1, 2, 3]), np.array([4, 5, 6]))
        with pytest.raises(EmptyArrayError):
            sum1(np.array([1, 2]), np.array([]))
        with pytest.raises(EmptyArrayError):
            sum1(np.array([]), np.array([3, 4]))
        with pytest.raises(UnexpectedDimensionError):
            sum1(np.array([[1, 2], [3, 4]]), np.array([5, 6, 7, 8]))
        with pytest.raises(UnexpectedDimensionError):
            sum1(np.array([1, 2, 3, 4]), np.array([[5, 6], [7, 8]]))

        @expected_arrays(non_empty=[False, True], dim=[2, 1])
        def sum2(a, b):
            return np.sum(a + b)
        assert 32 == sum2(np.array([[1, 2], [3, 4]]), np.array([5, 6]))
        with pytest.raises(ValueError):
            sum2(np.array([[], []]), np.array([5, 6]))
        with pytest.raises(EmptyArrayError):
            sum2(np.array([[1, 2], [3, 4]]), np.array([]))
        with pytest.raises(UnexpectedDimensionError):
            sum2(np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
        with pytest.raises(UnexpectedDimensionError):
            sum2(np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]))
