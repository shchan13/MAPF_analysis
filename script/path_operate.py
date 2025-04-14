# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from pandas import Series


def x_operate(row:Series, cfg:Dict):
    return row['runtime']
    # return row['time_gen']

def y_operate(row:Series, cfg:Dict):
    # return row['cost'] / row['global_lb']
    return row['global_lb'] - cfg['init_lb']
    # return row['dist_togo']
