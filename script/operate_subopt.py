# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from pandas import Series
from math import inf

def x_operate(row:Series, cfg:Dict):
    # return (row['lowerbound'] - row['root lb']) / row['root lb']
    if row['solution cost'] >= 0 and row['runtime'] <= cfg['time_limit']:
        return row['subopt']
        # return row['solution cost']
        # return row['lowerbound']
    return inf

def y_operate(row:Series, cfg:Dict):
    # return (row['lowerbound'] - row['root lb']) / row['root lb']
    if row['solution cost'] >= 0 and row['runtime'] <= cfg['time_limit']:
        return row['subopt']
        # return row['solution cost']
        # return row['lowerbound']
    return inf
