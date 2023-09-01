# -*- coding: UTF-8 -*-
'''Iteration processor'''

import os
import argparse
from typing import Dict, List
import yaml
import matplotlib.pyplot as plt
import util

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
        self.results:List = []


    def get_iter_val(self):
        for curr in self.config['files']:
            data_frame = util.read_file(curr['path'])
            curr['data'] = data_frame
            self.results.append(curr)


    def plot_fig(self):
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
            mf_color= 'white' if 'markerfacecolor' not in rst.keys() \
                else rst['markerfacecolor']
            zord = 0 if 'zorder' not in rst.keys() else rst['zorder']

            cur_x_pos = []
            if self.config['x_labels'] == 'iteration':
                cur_x_pos = [x + plt_rng*rid for x in range(1, max_x_len+1)]
            else:
                cur_x_pos = rst['data'][self.config['x_labels']]

            axs.plot(cur_x_pos, rst['data'][self.config['y_labels']],
                     label=rst['label'],
                     linewidth=self.config['line_width'],
                     marker=rst['marker'],
                     ms=self.config['marker_size'],
                     markerfacecolor=mf_color,
                     markeredgewidth=self.config['marker_width'],
                     color=rst['color'],
                     alpha=self.config['alpha'],
                     zorder=zord)

        # x_labels = axs.axes.get_xticks()
        # x_labels = [int(x) for x in x_labels]
        # axs.axes.set_xticklabels(x_labels, fontsize=self.config['text_size'])

        # y_labels = axs.axes.get_yticks()
        # y_labels = [int(y) for y in y_labels]
        # axs.axes.set_yticklabels(y_labels, fontsize=self.config['text_size'])
        plt.xticks(fontsize=self.config['text_size'])
        plt.xlabel(self.config['x_labels'], fontsize=self.config['text_size'])
        plt.yticks(fontsize=self.config['text_size'])
        plt.ylabel(self.config['y_labels'], fontsize=self.config['text_size'])
        plt.legend(loc="best", fontsize=self.config['text_size'])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    iter_proc = IterProcessor(args.config)
    iter_proc.get_iter_val()
    iter_proc.plot_fig()
