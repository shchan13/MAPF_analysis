# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from math import inf
from pandas import Series


def x_operate(row:Series, cfg:Dict):
    return row['runtime']

def y_operate(row:Series, cfg:Dict):
    return row['cost'] / row['global_lb']
