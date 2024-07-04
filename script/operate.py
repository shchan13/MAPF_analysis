# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

import numpy as np


def x_operate(row):
    return row['solution cost'] if row['solution cost'] > -1 else np.inf


def y_operate(row):
    return 1 if row['solution cost'] > -1 else 0  # number of success instances
