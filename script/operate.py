# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict

from pandas import Series
import numpy as np


def x_operate(row:Series, cfg:Dict):
    if row['solution cost'] > -1 and row['runtime'] <= cfg['time_limit']:
        return row['solution cost']
    return np.inf


def y_operate(row:Series, cfg:Dict):
    # return 1 if row['solution cost'] > -1 else 0  # number of success instances
    if row['solution cost'] > -1 and row['runtime'] + row['runtime build HWY'] <= cfg['time_limit']:
        return 1
    return 0
