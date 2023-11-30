# -*- coding: UTF-8 -*-

import logging
import os
import sys
import argparse
from typing import Dict, List
import yaml
import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd


class GetMin:
    def __init__(self):
        config_dir = '/home/rdaneel/MAPF_analysis/local/get_min.yaml'
        with open(config_dir, encoding='utf-8', mode='r') as fin:
            self.config = yaml.load(fin, Loader=yaml.FullLoader)

    def main(self):
        for cur_map in self.config['maps']:
            for scen in cur_map['scens']:
                for ag_num in cur_map['num_of_agents']:
                    data_frames = []
                    for solver in self.config['solvers']:
                        cur_df = util.get_csv_instance(self.config['exp_path'],
                                                       cur_map['name'],
                                                       scen,
                                                       ag_num,
                                                       solver['name'],
                                                       solver['dir_name'])
                        data_frames.append(cur_df)

                    out_df = pd.DataFrame(columns=data_frames[0].columns)
                    for insID in range(self.config['ins_num']):
                        out_ins = {}

                        out_ins['runtime'] = -1
                        for df in data_frames:
                            if out_ins['runtime'] < df.iloc[insID]['runtime']:
                                out_ins['runtime'] = df.iloc[insID]['runtime']

                        out_ins['solution cost'] = np.inf
                        best_id = -1
                        for dfID, df in enumerate(data_frames):
                            if df.iloc[insID]['solution cost'] < out_ins['solution cost']:
                                out_ins['solution cost'] = df.iloc[insID]['solution cost']
                                best_id = dfID

                        out_ins['initial solution cost'] = data_frames[best_id].iloc[insID]['initial solution cost']
                        out_ins['lower bound'] = data_frames[best_id].iloc[insID]['lower bound']
                        out_ins['sum of distance'] =\
                            data_frames[best_id].iloc[insID]['sum of distance']

                        out_ins['makespan'] = np.inf
                        for df in data_frames:
                            if df.iloc[insID]['makespan'] < out_ins['makespan']:
                                out_ins['makespan'] = df.iloc[insID]['makespan']

                        out_ins['iterations'] = 0
                        for df in data_frames:
                            out_ins['iterations'] += df.iloc[insID]['iterations']
                        out_ins['succ iterations'] = 0
                        for df in data_frames:
                            out_ins['succ iterations'] += df.iloc[insID]['succ iterations']
                        out_ins['failed iterations'] = 0
                        for df in data_frames:
                            out_ins['failed iterations'] += df.iloc[insID]['failed iterations']
                        out_ins['depth'] = data_frames[best_id].iloc[insID]['succ iterations']

                        out_ins['group size'] = 0
                        for df in data_frames:
                            out_ins['group size'] += df.iloc[insID]['group size']
                        out_ins['group size'] /= len(data_frames)

                        out_ins['runtime of initial solution'] =\
                            data_frames[best_id].iloc[insID]['runtime of initial solution']

                        out_ins['restart times'] = 0
                        for df in data_frames:
                            out_ins['restart times'] += df.iloc[insID]['restart times']

                        out_ins['area under curve'] =\
                            data_frames[best_id].iloc[insID]['area under curve']

                        out_ins['LL expanded nodes'] = 0
                        for df in data_frames:
                            out_ins['LL expanded nodes'] += df.iloc[insID]['LL expanded nodes']
                        out_ins['LL generated'] = 0
                        for df in data_frames:
                            out_ins['LL generated'] += df.iloc[insID]['LL generated']
                        out_ins['LL reopened'] = 0
                        for df in data_frames:
                            out_ins['LL reopened'] += df.iloc[insID]['LL reopened']
                        out_ins['LL runs'] = 0
                        for df in data_frames:
                            out_ins['LL runs'] += df.iloc[insID]['LL runs']

                        out_ins['preprocessing runtime'] =\
                            data_frames[best_id].iloc[insID]['preprocessing runtime']

                        out_ins['vm usage'] = 0
                        for df in data_frames:
                            out_ins['vm usage'] += df.iloc[insID]['vm usage']
                        out_ins[' rss usage'] = 0
                        for df in data_frames:
                            out_ins[' rss usage'] += df.iloc[insID][' rss usage']

                        out_ins['solver name'] = data_frames[best_id].iloc[insID]['solver name']
                        out_ins['instance name'] = data_frames[best_id].iloc[insID]['instance name']

                        df_dictionary = pd.DataFrame([out_ins])
                        out_df = pd.concat([out_df, df_dictionary], ignore_index=True)

                    out_dir = '/home/rdaneel/my_exp2/'+ cur_map['name'] + '/DETACHED'
                    os.makedirs(out_dir, exist_ok=True)
                    out_filename = cur_map['name'] + '-random-' + str(ag_num) + '-DETACHED_16_4.csv'
                    out_df.to_csv(os.path.join(out_dir,out_filename), index=False)


if __name__ == '__main__':
    getM = GetMin()
    getM.main()
