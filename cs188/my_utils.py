"""
Utility functions for all projects.
"""
from typing import Sequence


def avg(nums):
    return sum(nums) / len(nums)


def getIndexOfMax(values: Sequence, default=-1):
    return max(range(len(values)), key=values.__getitem__, default=default)


def getIndexOfMin(values: Sequence, default=-1):
    return min(range(len(values)), key=values.__getitem__, default=default)
