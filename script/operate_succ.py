# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from math import inf
from pandas import Series


def x_operate(row:Series, cfg:Dict):
    if row['solution cost'] > -1 and row['runtime'] <= cfg['time_limit']:
        return row['solution cost']
    return inf

def y_operate(row:Series, cfg:Dict):
    runtime_hwy = row['runtime build HWY']
    if row['runtime build HWY'] == 1.79769E+308:
        runtime_hwy = 0
    if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
        return 1
    return 0
