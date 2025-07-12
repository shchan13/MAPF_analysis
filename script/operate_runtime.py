# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from pandas import Series

def x_operate(row:Series, cfg:Dict):
    return min(row['runtime'], cfg['time_limit'])

def y_operate(row:Series, cfg:Dict):
    # return min(row['runtime'], cfg['time_limit'])
    return min(row['runtime build HWY'], cfg['time_limit'])
    # return min(row['runtime'] + row['runtime build HWY'], cfg['time_limit'])
