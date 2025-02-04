# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from math import inf
from pandas import Series


def x_operate(row:Series, cfg:Dict):
    if row['solution cost'] > -1 and row['runtime'] <= cfg['time_limit']:
        return row['solution cost']
    return inf

# def y_operate(row:Series, cfg:Dict):
#     runtime_hwy = row['runtime build HWY']
#     # if row['runtime build HWY'] == 1.79769E+308:
#     #     runtime_hwy = 0
#     # if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
#     #     return 1
#     # return 0
#     if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
#         # return row['solution cost']
#         return row['solution cost'] / row['lowerbound']
#     return inf

def y_operate(row:Series, cfg:Dict):
    # return row['lowerbound']
    # return (row['lowerbound'] - row['root lb']) / (row['cleanup'] + 1)
    # return row['cleanup'] + 1
    # return row['runtime fval']
    # return row['root target conf']
    return row['root conf']
    # return row['ll gen']
    # return row['ll exp']
    # return row['ll search']
    # return row['hl exp']
    # return row['root cost']
    # return min(row['runtime build HWY'], cfg['time_limit'])
    # return row['runtime build HWY']
    # return min(row['runtime'], cfg['time_limit'])
    # runtime_hwy = row['runtime build HWY']
    # if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
    #     # return row['solution cost'] - row['root cost']
    #     # return row['runtime']
    #     return row['solution cost']
    #     # return (row['lowerbound'] - row['root lb']) / row['hl gen cleanup']
    # return inf
