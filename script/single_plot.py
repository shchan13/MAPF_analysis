# -*- coding: UTF-8 -*-
"""
Plot a single figure
"""

import argparse
import logging
import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from math import inf
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import util
import yaml


class MAPFPlotter:
    """Plotter for results
    The results should be in the .csv format.
    """
    def __init__(self, in_cfg) -> None:
        self.cfg:Dict = {}  # Configuration
        self.rst:Dict = {}  # Results

        cfg_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), in_cfg)
        with open(cfg_dir, encoding='utf-8', mode='r') as fin:
            self.cfg = yaml.load(fin, Loader=yaml.FullLoader)

        if 'ins_num' not in self.cfg.keys():
            self.cfg['ins_num'] = inf

        # Create a module spec from the given path in configuration
        self.spec = spec_from_file_location('operate', self.cfg['y_axis']['script'])

        self.func = module_from_spec(self.spec)  # Load the module from the spec
        sys.modules['operate'] = self.func  # Add the module to sys.modules

        # Execute the module to make its attributes accessible
        self.spec.loader.exec_module(self.func)

        # Create the folder for self.cfg['output_dir']
        if not os.path.isdir(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])


    def get_val(self):
        """Get the data from each row according to the y axis.
        Compute the average, standard deviation, and confidence interval
        of the obtained data.
        """
        for p in self.cfg['plots']:
            assert len(self.cfg['x_axis']['range']) == len(p['data'])
            self.rst[p['label']] = {}
            for it_id, it in enumerate(p['data']):
                cur_x = self.cfg['x_axis']['range'][it_id]
                self.rst[p['label']][cur_x] = { 'data': [],
                                                'avg': 0,
                                                'std': 0.0,
                                                'ci': 0.0 }
                for fin in it:  # Iterate over files per x-axis
                    prev_len = len(self.rst[p['label']][cur_x]['data'])
                    df = util.read_file(fin)
                    for row_id, row in df.iterrows():
                        if row_id >= self.cfg['ins_num']:
                            break
                        row_val = self.func.y_operate(row, self.cfg)
                        self.rst[p['label']][cur_x]['data'].append(row_val)
                    new_data_num = len(self.rst[p['label']][cur_x]['data']) - prev_len
                    if new_data_num < self.cfg['ins_num']:
                        logging.warning('%s does not match the instance number', fin)

                # Compute statistic data ignoring inf
                effective_data = []  # Operated data without inf
                for val in self.rst[p['label']][cur_x]['data']:
                    if val is not inf:
                        self.rst[p['label']][cur_x]['avg'] += val
                        effective_data.append(val)
                if len(effective_data) > 0:
                    self.rst[p['label']][cur_x]['avg'] /= len(effective_data)
                    self.rst[p['label']][cur_x]['std'] = np.std(effective_data)
                    self.rst[p['label']][cur_x]['ci'] =\
                        1.96 * self.rst[p['label']][cur_x]['std'] / np.sqrt(len(effective_data))
                print(p['label'], ',', cur_x, ': ', self.rst[p['label']][cur_x]['avg'])


    def plot_fig(self):
        """Plot function for customize x and y axes.
        Save the figure in the .svg format
        """
        plt.figure(figsize=(self.cfg['fig_width'], self.cfg['fig_height']))
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        left_bd = -1 * self.cfg['set_shift']
        right_bd = self.cfg['set_shift']
        plt_rng = (right_bd - left_bd) / len(self.cfg['plots'])
        x_num = range(1, len(self.cfg['x_axis']['range'])+1)

        for pid, p in enumerate(self.cfg['plots']):
            x = [n + plt_rng * pid for n in x_num]

            val, dev = [], []
            for it_id in range(len(p['data'])):
                cur_x = self.cfg['x_axis']['range'][it_id]
                val.append(self.rst[p['label']][cur_x]['avg'])
                if self.cfg['is_std']:
                    dev.append(self.rst[p['label']][cur_x]['ci'])
                elif self.cfg['is_ci']:
                    dev.append(self.rst[p['label']][cur_x]['std'])

            if len(dev) > 0:
                plt.errorbar(x, val, yerr=dev,
                    label=p['label'],
                    color=p['color'],
                    marker=p['marker'],
                    zorder=p['zorder'],
                    alpha=self.cfg['alpha'],
                    markerfacecolor=p['markerfacecolor'],
                    linewidth=self.cfg['line_width'],
                    markeredgewidth=self.cfg['marker_width'],
                    ms=self.cfg['marker_size'])
            else:
                plt.plot(x, val,
                    label=p['label'],
                    color=p['color'],
                    marker=p['marker'],
                    zorder=p['zorder'],
                    alpha=self.cfg['alpha'],
                    markerfacecolor=p['markerfacecolor'],
                    linewidth=self.cfg['line_width'],
                    markeredgewidth=self.cfg['marker_width'],
                    ms=self.cfg['marker_size'])

        plt.xticks(x_num,
                   labels=self.cfg['x_axis']['range'],
                   fontsize=self.cfg['text_size']['x_axis'])
        plt.xlabel(self.cfg['x_axis']['label'],
                   fontsize=self.cfg['text_size']['x_axis'])

        plt.yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
                   labels=self.cfg['y_axis']['range'],
                   fontsize=self.cfg['text_size']['y_axis'])
        plt.ylabel(self.cfg['y_axis']['label'],
                   fontsize=self.cfg['text_size']['y_axis'])

        plt.tight_layout()
        plt.grid(axis='y')
        plt.legend(fontsize=self.cfg['text_size']['legend'])  # markerscale=0.7
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']))
        plt.show()


    def plot_fig_instance(self):
        """Plot the row values instance by instance
        """

        # Remove the line and decrease the alpha before plotting
        self.cfg['line_width'] = 0.0
        self.cfg['marker_size'] = 5.0
        self.cfg['alpha'] = 0.8

        plt.figure(figsize=(self.cfg['fig_width'], self.cfg['fig_height']))
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        x_num = range(1, len(self.cfg['x_axis']['range']) * self.cfg['ins_num'] +1)
        for p in self.cfg['plots']:
            val = []
            for it_id in range(len(p['data'])):
                cur_x = self.cfg['x_axis']['range'][it_id]
                for y in self.rst[p['label']][cur_x]['data']:
                    val.append(y)
            plt.plot(x_num, val,
                     label=p['label'],
                     color=p['color'],
                     marker=p['marker'],
                     zorder=p['zorder'],
                     alpha=self.cfg['alpha'],
                     markerfacecolor=p['markerfacecolor'],
                     linewidth=self.cfg['line_width'],
                     markeredgewidth=self.cfg['marker_width'],
                     ms=self.cfg['marker_size'])
        xticks_list = [1]
        for i in range(len(self.cfg['x_axis']['range'])):
            xticks_list.append((i + 1) * self.cfg['ins_num'])
        plt.xticks(xticks_list,
                   labels=xticks_list,
                   fontsize=self.cfg['text_size']['x_axis'])
        plt.xlabel(self.cfg['x_axis']['label'],
                   fontsize=self.cfg['text_size']['x_axis'])

        plt.yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
                   labels=self.cfg['y_axis']['range'],
                   fontsize=self.cfg['text_size']['y_axis'])
        plt.ylabel(self.cfg['y_axis']['label'],
                   fontsize=self.cfg['text_size']['y_axis'])

        plt.tight_layout()
        plt.grid(axis='y')
        plt.legend(fontsize=self.cfg['text_size']['title'])
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input')
    parser.add_argument('--config', type=str, default='single_plot.yaml')
    args = parser.parse_args()

    mapf_plotter = MAPFPlotter(args.config)
    mapf_plotter.get_val()
    mapf_plotter.plot_fig()
    # mapf_plotter.plot_fig_instance()
