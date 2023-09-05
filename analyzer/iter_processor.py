# -*- coding: UTF-8 -*-
'''Iteration processor'''

import os
import argparse
from typing import Dict, List
import tqdm
import yaml
import matplotlib.pyplot as plt
import util
import numpy as np

DESTROY_STRATEGY = {0: 'Random', 1: 'Agent-based', 2: 'Intersection-based'}

class IterProcessor:
    def __init__(self, in_config) -> None:
        self.config: Dict = {}
        config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  in_config)
        with open(config_dir, encoding='utf-8', mode='r') as fin:
            self.config = yaml.load(fin, Loader=yaml.FullLoader)

        self.files:str = self.config['files']  # [{path, label, color,...}, ...]
        self.x_labels:str = self.config['x_labels']
        self.y_labels:str = self.config['y_labels']
        self.results:List = []  # Each element is a dataframe from the files


    def get_iter_val(self):
        for curr in self.config['files']:
            data_frame = util.read_file(curr['path'])
            curr['data'] = data_frame
            self.results.append(curr)
        if self.config['y_labels'] == 'destroy weight':
            self.proc_multi_val('destroy weight')
        if self.config['y_labels'] == 'destroy probability':
            self.proc_multi_val('destroy probability')


    def proc_multi_val(self, col_name:str):
        """ Process data with multiple values separated by '_'

        Args:
            col_name (str): the name of the column in the dataframe
        """
        print('Processing multiple values... ', end='')
        for rst in self.results:
            if col_name not in rst['data'].columns:
                continue
            qbar = tqdm.tqdm(total=len(rst['data'][col_name]), desc=rst['label'])
            for idx, val in enumerate(rst['data'][col_name]):
                val = [float(ele) for ele in val.split('_')]
                rst['data'][col_name].at[idx] = val
                qbar.update(1)
            qbar.close()


    def plot_fig(self):
        print('Plot figure... ', end='')
        plt.close('all')
        _, axs = plt.subplots(nrows=1, ncols=1,
                              figsize=(self.config['figure_width'],
                                       self.config['figure_height']),
                              dpi=80, facecolor='w', edgecolor='k')
        max_x_len = 0
        for rst in self.results:
            if len(rst['data']) > max_x_len:
                max_x_len = len(rst['data'])

        left_bd  = -1 * self.config['set_shift']
        right_bd = self.config['set_shift']
        plt_rng  = (right_bd - left_bd) / len(self.results)

        for rid,rst in enumerate(self.results):
            mf_color= 'white' if 'markerfacecolor' not in rst.keys() else rst['markerfacecolor']
            zord = 0 if 'zorder' not in rst.keys() else rst['zorder']

            x_pos = []
            if self.config['x_labels'] == 'iteration':
                x_pos = [x + plt_rng*rid for x in range(1, max_x_len+1)]
            else:
                x_pos = rst['data'][self.config['x_labels']].to_list()

            y_pos = rst['data'][self.config['y_labels']]
            if self.config['y_labels'] in ['destroy probability', 'destroy weight']:
                x_pos.insert(0, 0.0)
                widths = [x_pos[ii] - x_pos[ii-1] for ii in range(1,len(x_pos))]
                x_pos.pop(-1)
                bottom = [0 for _ in x_pos]
                for idx in range(len(y_pos.iloc[0])):
                    cur_y:List = []
                    for yval in y_pos:
                        cur_y.append(yval[idx])
                    axs.bar(x_pos, cur_y, width=widths, align='edge', bottom=bottom,
                            label=DESTROY_STRATEGY[idx])
                    bottom = [bottom[j] + cur_y[j] for j in range(len(bottom))]
            else:
                axs.plot(x_pos, y_pos,
                    label=rst['label'],
                    linewidth=self.config['line_width'],
                    marker=rst['marker'],
                    ms=self.config['marker_size'],
                    markerfacecolor=mf_color,
                    markeredgewidth=self.config['marker_width'],
                    color=rst['color'],
                    alpha=self.config['alpha'],
                    zorder=zord)

        # x_labels = [int(x) for x in x_labels]
        # axs.axes.set_xticklabels(x_labels, fontsize=self.config['text_size'])

        # y_labels = axs.axes.get_yticks()
        # y_labels = [int(y) for y in y_labels]
        # axs.axes.set_yticklabels(y_labels, fontsize=self.config['text_size'])
        plt.xticks(fontsize=self.config['text_size'])
        plt.xlabel(self.config['x_labels'].capitalize(),
                   fontsize=self.config['text_size'])

        if self.config['y_labels'] in ['destroy probability', 'destroy weight']:
            plt.yticks(np.arange(0, 1.01, 0.2))
        plt.yticks(fontsize=self.config['text_size'])
        plt.ylabel(self.config['y_labels'].capitalize(),
                   fontsize=self.config['text_size'])

        if self.config['y_labels'] in ['destroy probability', 'destroy weight']:
            handles, labels = plt.gca().get_legend_handles_labels()
            order = list(DESTROY_STRATEGY.keys())
            order.reverse()
            plt.legend([handles[i] for i in order], [labels[i] for i in order],
                       loc="best", fontsize=self.config['text_size'])
        else:
            plt.legend(loc="best", fontsize=self.config['text_size'])
        plt.savefig(self.config['save_path'])
        # plt.show()
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    iter_proc = IterProcessor(args.config)
    iter_proc.get_iter_val()
    iter_proc.plot_fig()
