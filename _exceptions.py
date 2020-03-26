#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Error(Exception):
    pass


class EmptyArrayError(Error):
    def __str__(self):
        return 'Array is empty.'


class UnexpectedDimensionError(Error):
    def __init__(self, expected_dim, dim):
        self._expected_dim = expected_dim
        self._dim = dim

    def __str__(self):
        return 'Dimension should be {}, but it is {}.'.format(self._expected_dim, self._dim)


class DecoratorArgumentError(Error):
    pass
