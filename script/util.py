# -*- coding: UTF-8 -*-
"""Utility functions"""

import logging
import os
import sys
import random
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

LARGE_MAPS:List[str] = ['den520d',
                        'warehouse-10-20-10-2-1',
                        'warehouse-20-40-10-2-1',
                        'warehouse-20-40-10-2-2']
ANYTIME_SOLVERS:List[str] = ['LACAMLNS', 'AnytimeCBS']
FIG_AXS:Dict[int, Tuple[int,int]] = {1: (1,1),
                                     2: (1,2),
                                     3: (1,3),
                                     4: (2,2),
                                     5: (1,5),
                                     6: (2,3),
                                     8: (2,4),
                                     9: (3,3)}
MAX_LABEL_NUM = 5
LABEL_SCALE = 1000


def read_file(in_path:str) -> pd.DataFrame:
    """ Read the csv file with pandas

    Args:
        in_path (str): path to the csv file

    Returns:
        pd.DataFrame: the csv file
    """
    if not os.path.exists(in_path):
        logging.error('%s does not exist!', in_path)
        sys.exit()
    else:
        return pd.read_csv(in_path, low_memory=False)


def get_file_dir(exp_path:str, map_name:str, solver_name:str) -> str:
    """Get the path to the csv files

    Args:
        exp_path (str): path to the whole experiments
        map_name (str): map name
        solver_name (str): solver name

    Returns:
        str: path to the csv files
    """
    map_dir = os.path.join(exp_path, map_name)
    out_dir = os.path.join(map_dir, solver_name)
    return out_dir


def get_file_name(map_name:str, scen:str, ag_num:int, solver_name: str) -> str:
    """Get the name of the csv file (end with .csv)

    Args:
        map_name (str): map_name
        scen (str): even or random scen
        ag_num (int): number of agents
        solver_name (str): the solver name
    Returns:
        str: name of the csv files
    """
    out_name = map_name + '-' + scen + '-' + str(ag_num) + '-' + solver_name + '.csv'
    return out_name


def get_csv_instance(exp_path:str, map_name:str, scen:str, ag_num:int,
                     solver_name:str, solver_dir_name:str=None):
    """Get the path and read the csv with pandas

    Args:
        map_name (str): map_name
        scen (str): even or random scen
        ag_num (int): number of agents
        solver_name (str): the solver name from config.yaml
        solver_dir_name (str): the name of the directory where the solver saves
    Returns:
        pd.DataFrame: the csv file
    """
    if solver_dir_name is None:
        solver_dir_name = solver_name
    return read_file(os.path.join(
        get_file_dir(exp_path, map_name, solver_dir_name),
        get_file_name(map_name, scen, ag_num, solver_name)))


def create_csv_file(exp_path:str, map_name:str, scen:str, ag_num:int, ins_num:int, sol_dir:str,
                    sol_names:List[str], mode:str='min', objective:str='runtime'):
    csv_files = {}
    first_name = ""
    for idx, _name_ in enumerate(sol_names):
        csv_files[_name_] = get_csv_instance(exp_path, map_name, scen, ag_num, _name_, sol_dir)
        if idx == 0:
            first_name = _name_

    # Sort the csv_files according to the objective
    buffer = {}
    for col in csv_files[first_name].columns:
        buffer[col] = []

    target_idx = -np.inf
    if mode == 'min':
        target_idx = 0
    elif mode == 'mid':
        target_idx = len(sol_names)//2 - 1
    elif mode == 'max':
        target_idx = len(sol_names) - 1

    for idx in range(ins_num):
        tmp_rows = {}
        tmp_objs = {}
        for _name_, _file_ in csv_files.items():
            row = _file_.iloc[idx]
            tmp_rows[_name_] = row
            tmp_objs[_name_] = row[objective]

        sorted_objs = dict(sorted(tmp_objs.items(), key=lambda item : item[1]))
        for j, val in enumerate(sorted_objs.items()):
            if j == target_idx:
                tmp_row_val = tmp_rows[val[0]]
                for k, _ in enumerate(tmp_row_val.items()):
                    buffer[k].append(tmp_row_val[k])
                break

    solver_type = sol_names[0].split('_')[0]
    out_dir = exp_path + map_name + '/' + solver_type + '_' + mode + '_' + objective
    if not os.path.exists(out_dir):  # Create a new directory because it does not exist
        os.makedirs(out_dir)

    out_file_name = map_name + '-' + scen + '-' + str(ag_num) + '-' + \
        solver_type + '_' + mode + '_' + objective + '.csv'
    out_df = pd.DataFrame(buffer)
    out_df.to_csv(path_or_buf=os.path.join(out_dir, out_file_name), index=False)


def process_val(raw_value, raw_index:str, solution_cost:int,
                runtime:float, time_limit:float,
                solver_name:str, succ_only:bool=False):
    is_anytime = solver_name in ANYTIME_SOLVERS
    is_succ = solution_cost >= 0 and (runtime <= time_limit or is_anytime)

    # if solver_name == 'SYNCLNS-LACAM2-PP_16_8' and raw_index == 'iterations':
        # return raw_value * 8

    if raw_index  == 'succ':
        return int(is_succ)

    if raw_index in ['runtime', 'runtime of initial solution']:
        return min(max(raw_value, 0), time_limit)

    if raw_index in ['num_total_conf', 'num_0child'] and raw_value == 0:
        return np.inf

    if raw_index == 'solution cost' and raw_value < 0:
        return np.inf

    if succ_only and not is_succ:
        return np.inf

    assert raw_value >= 0
    return raw_value


def load_map(map_file:str):
    """load map from the map_file
        Args:
            map_file (str): file of the map
    """
    height = -1
    width = -1
    out_map = []
    num_freespace = 0
    with open(map_file, mode="r", encoding="UTF-8") as fin:
        fin.readline()  # Skip the first line
        height = int(fin.readline().strip("\n").split(" ")[-1])
        width  = int(fin.readline().strip("\n").split(" ")[-1])
        fin.readline()  # Skip the line "map"
        for line in fin.readlines():
            line = list(line.strip("\n"))
            out_line = [_char_ == "." for _char_ in line]
            out_map.append(out_line)
            num_freespace += sum(bool(x) for x in out_line)

    return height, width, out_map, num_freespace


def load_instance(ins_file:str, agent_num:int=-1):
    """load the MAPF instance from ins_file

    Args:
        ins_file (str): file path of the MAPF instance

    Returns:
        Dict: start and goal locations of all agents
    """
    ins_locs:Dict[str:List[tuple[int,int]]] = {"start": [], "goal": []}
    with open(ins_file, mode="r", encoding="UTF-8") as fin:
        fin.readline()  # skip version line
        for idx, line in enumerate(fin.readlines()):
            if idx == agent_num:  # idx equals the number of agents in the list
                break
            line = line.strip("\n").split("\t")
            start_col = int(line[4])
            start_row = int(line[5])
            goal_col = int(line[6])
            goal_row = int(line[7])
            ins_locs["start"].append((start_row, start_col))
            ins_locs["goal"].append((goal_row, goal_col))
    assert len(ins_locs["start"]) == len(ins_locs["goal"])
    assert len(ins_locs["start"]) <= agent_num
    return ins_locs


def get_map_name(map_file:str):
    """Get the map name from the map_file

    Args:
        map_file (str): the path to the map
    """
    return map_file.split('/')[-1].split('.')[0]


def random_walk(in_map:List[List[bool]], init_loc:Tuple, steps:int):
    """Random walk from the init_loc on in_map with steps

    Args:
        in_map (List[List[bool]]): map
        init_loc (Tuple): initial location of the agent
        steps (int): number of steps to move
    """
    if in_map[init_loc[0]][init_loc[1]] is False:
        logging.error("location (%d,%d) should be a free space!", init_loc[0], init_loc[1])
        sys.exit()

    curr_loc = init_loc
    height = len(in_map)
    width = len(in_map[0])
    for _ in range(steps):
        next_locs = [(curr_loc[0]+1, curr_loc[1]),
                     (curr_loc[0]-1, curr_loc[1]),
                     (curr_loc[0], curr_loc[1]+1),
                     (curr_loc[0], curr_loc[1]-1)]
        random.shuffle(next_locs)

        for next_loc in next_locs:
            if  -1 < next_loc[0] < height and\
                -1 < next_loc[1] < width and\
                in_map[next_loc[0]][next_loc[1]] is True:
                curr_loc = next_loc
                break
    return curr_loc
