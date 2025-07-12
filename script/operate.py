# -*- coding: UTF-8 -*-
"""Custom operation for each row"""

from typing import Dict
from math import inf
from pandas import Series


def x_operate(row:Series, cfg:Dict):
    # if row['solution cost'] > -1 and row['runtime'] <= cfg['time_limit']:
    #     return row['solution cost']
    # return inf
    return row['ll search']
    # return min(row['runtime'], cfg['time_limit'])
    # return row['ll exp']

# def y_operate(row:Series, cfg:Dict):
#     runtime_hwy = row['runtime build HWY']
#     if row['runtime build HWY'] == 1.79769E+308:
#         runtime_hwy = 0
#     # if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
#     #     return 1
#     # return 0
#     if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
#         return row['solution cost'] / row['lowerbound']
#     return inf

def y_operate(row:Series, cfg:Dict):
    # return row['hl exp'] / row['hl gen']
    # return row['lowerbound']
    # return row['lowerbound'] - row['root lb']
    return (row['lowerbound'] - row['root lb']) / row['root lb']  # LBI
    # return (row['lowerbound'] - row['root lb']) / (row['hl focal'] + row['hl open'] + row['hl cleanup'])
    # return (row['root conf'] - row['remaining conf']) / (row['hl depth'])
    # return row['hl gen focal'] / (row['hl gen focal'] + row['hl gen cleanup'])
    # return row['hl cleanup'] / (row['hl focal'] + row['hl open'] + row['hl cleanup'])
    # return row['ll search'] + row['#eff agents']
    # return (row['root conf'] - row['remaining conf']) / row['hl gen']
    # assert row['hl exp'] > 0
    # assert row['hl depth'] > 0
    # return (row['hl depth'] + 1) / row['hl exp']
    # return row['hl depth']
    # return row['root cost']
    # return min(row['runtime build HWY'], cfg['time_limit'])
    ##############################################
    # runtime_hwy = row['runtime build HWY']
    # if row['runtime build HWY'] == 1.79769E+308:
    #     runtime_hwy = 0
    # # # return row['runtime build HWY']
    # # # return min(row['runtime'], cfg['time_limit'])
    # # if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:
    # #     return row['runtime'] + runtime_hwy
    # # return cfg['time_limit']
    # if row['solution cost'] > -1 and row['runtime'] + runtime_hwy <= cfg['time_limit']:  # succ only
    #     if row['ll search'] == 0:
    #         return 0
    #     return row['ll exp'] / row['ll search']
    # return inf
