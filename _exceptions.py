#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class BaseError(Exception):
    pass


class EmptyArrayError(BaseError):
    def __str__(self):
        return 'Array is empty.'


class DimensionError(BaseError):
    def __init__(self, dim):
        self._dim = dim

    def __str__(self):
        return 'Dimension should be {}.'.format(self._dim)
