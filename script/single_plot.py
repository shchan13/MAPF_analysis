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

import numpy as np
import util
import yaml
import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter


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


    def get_scatter_val(self):
        """Get the (x,y) value for each data
        """
        for p in self.cfg['plots']:
            # assert len(self.cfg['x_axis']['range']) == len(p['data'])
            self.rst[p['label']] = {'x': [], 'y': []}
            for it in p['data']:
                for fin in it:
                    assert len(self.rst[p['label']]['y']) == len(self.rst[p['label']]['x'])
                    prev_len = len(self.rst[p['label']]['y'])
                    df = util.read_file(fin)
                    for row_id, row in df.iterrows():
                        if row_id >= self.cfg['ins_num']:
                            break
                        row_xval = self.func.x_operate(row, self.cfg)
                        row_yval = self.func.y_operate(row, self.cfg)
                        self.rst[p['label']]['x'].append(row_xval)
                        self.rst[p['label']]['y'].append(row_yval)
                    new_data_num = len(self.rst[p['label']]['y']) - prev_len
                    if new_data_num < self.cfg['ins_num']:
                        logging.warning('%s does not match the instance number', fin)

    def get_scatter_comparison(self):
        """Get the (x,y) value between two algorithm
        """
        p = self.cfg['plots'][0]
        self.rst[p['label']] = {'data': []}
        for it in p['data']:
            for fin in it:
                df = util.read_file(fin)
                for row_id, row in df.iterrows():
                    if row_id >= self.cfg['ins_num']:
                        break
                    row_xval = self.func.y_operate(row, self.cfg)
                    self.rst[p['label']]['data'].append(row_xval)

        p = self.cfg['plots'][1]
        self.rst[p['label']] = {'data': []}
        for it in p['data']:
            for fin in it:
                df = util.read_file(fin)
                for row_id, row in df.iterrows():
                    if row_id >= self.cfg['ins_num']:
                        break
                    row_xval = self.func.y_operate(row, self.cfg)
                    self.rst[p['label']]['data'].append(row_xval)

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
                    linestyle=p['linestyle'],
                    linewidth=self.cfg['line_width'],
                    markeredgewidth=self.cfg['marker_width'],
                    ms=self.cfg['marker_size'])
            else:
                plt.plot(x, val,
                    label=p['label'],
                    color=p['color'],
                    marker=p['marker'],
                    zorder=p['zorder'],
                    alpha=p['alpha'],
                    markerfacecolor=p['markerfacecolor'],
                    linestyle=p['linestyle'],
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
        if self.cfg['y_grid']:
            plt.grid(axis='y')
        if self.cfg['set_legend']:
            plt.legend(fontsize=self.cfg['text_size']['legend'])  # markerscale=0.7
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']))
        plt.show()


    def plot_fig_instance(self):
        """Plot the row values instance by instance
        """

        # Remove the line and decrease the alpha before plotting
        self.cfg['line_width'] = 0.0
        self.cfg['marker_size'] = 5.0
        self.cfg['alpha'] = 0.6

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
        if self.cfg['set_legend']:
            plt.legend(fontsize=self.cfg['text_size']['title'])
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']))
        plt.show()

    def plot_scatter(self):
        """Plot the points according to the x-y values
        """

        # Remove the line and decrease the alpha before plotting
        # self.cfg['line_width'] = 0.0
        # self.cfg['marker_size'] = 5.0
        # self.cfg['alpha'] = 0.8
        self.cfg['line_width'] = 0.0
        self.cfg['marker_size'] = 4.0
        self.cfg['alpha'] = 0.5

        plt.figure(figsize=(self.cfg['fig_width'], self.cfg['fig_height']))
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        # for p in self.cfg['plots']:
        #     plt.plot(self.rst[p['label']]['x'], self.rst[p['label']]['y'],
        #              label=p['label'],
        #              color=p['color'],
        #              marker=p['marker'],
        #              zorder=p['zorder'],
        #              alpha=self.cfg['alpha'],
        #              markerfacecolor=p['markerfacecolor'],
        #              linewidth=self.cfg['line_width'],
        #              markeredgewidth=self.cfg['marker_width'],
        #              ms=self.cfg['marker_size'])

        for p in self.cfg['plots']:
            plt.plot(self.rst[p['label']]['x'][:125],
                     self.rst[p['label']]['y'][:125],
                     label='city',
                     color='orange',
                     marker='.',
                     zorder=3,
                     alpha=self.cfg['alpha'],
                     linewidth=self.cfg['line_width'],
                     ms=self.cfg['marker_size'])
            plt.plot(self.rst[p['label']]['x'][125:250],
                     self.rst[p['label']]['y'][125:250],
                     label='den520d',
                     color='purple',
                     marker='.',
                     zorder=2,
                     alpha=self.cfg['alpha'],
                     linewidth=self.cfg['line_width'],
                     ms=self.cfg['marker_size'])
            plt.plot(self.rst[p['label']]['x'][250:375],
                     self.rst[p['label']]['y'][250:375],
                     label='ost003d',
                     color='teal',
                     marker='.',
                     zorder=1,
                     alpha=self.cfg['alpha'],
                     linewidth=self.cfg['line_width'],
                     ms=self.cfg['marker_size'])
            plt.plot(self.rst[p['label']]['x'][375:500],
                     self.rst[p['label']]['y'][375:500],
                     label='warehouse',
                     color='dodgerblue',
                     marker='.',
                     zorder=0,
                     alpha=self.cfg['alpha'],
                     linewidth=self.cfg['line_width'],
                     ms=self.cfg['marker_size'])

        plt.xticks([r*self.cfg['x_axis']['scale'] for r in self.cfg['x_axis']['show_range']],
                   labels=self.cfg['x_axis']['show_range'],
                   fontsize=self.cfg['text_size']['x_axis'])
        plt.xlabel(self.cfg['x_axis']['label'],
                   fontsize=self.cfg['text_size']['x_axis'])

        plt.yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
                   labels=self.cfg['y_axis']['range'],
                   fontsize=self.cfg['text_size']['y_axis'])
        plt.ylabel(self.cfg['y_axis']['label'],
                   fontsize=self.cfg['text_size']['y_axis'])

        plt.tight_layout()
        if self.cfg['set_legend']:
            plt.legend(fontsize=self.cfg['text_size']['title'])
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']))
        plt.show()


    def plot_scatter_comparison(self):
        """Plot the points according to the x-y values
        """
        # Remove the line and decrease the alpha before plotting
        self.cfg['line_width'] = 0.0
        self.cfg['marker_size'] = 4.0
        self.cfg['alpha'] = 0.5

        plt.figure(figsize=(self.cfg['fig_width'], self.cfg['fig_height']))
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        yvals = np.linspace(self.cfg['linespace']['start'], self.cfg['linespace']['end'])
        xvals = yvals
        plt.plot(xvals, yvals,
                 label='1x',
                 color='grey',
                 linestyle=':',
                 linewidth=1.0)

        # xvals = yvals / 10.0
        # plt.plot(xvals, yvals,
        #          label='10x',
        #          color='grey',
        #          linestyle='--',
        #          linewidth=1.0)

        # xvals = yvals / 100.0
        # plt.plot(xvals, yvals,
        #          label='100x',
        #          color='grey',
        #          linestyle='-.',
        #          linewidth=1.0)

        ##### SOC comparison #####
        # invalid_cnt = 0
        # better_cnt = 0
        # worse_cnt = 0
        # valid_cnt = 0
        # for idx, val1 in enumerate(self.rst[self.cfg['plots'][1]['label']]['data']):
        #     val0 = self.rst[self.cfg['plots'][0]['label']]['data'][idx]
        #     if val0 == inf or val1 == inf:
        #         invalid_cnt += 1
        #         continue
        #     valid_cnt += 1
        #     if val1 < val0:
        #         better_cnt += 1
        #     elif val1 > val0:
        #         worse_cnt += 1
        # print('Better count: ', better_cnt)
        # print('Worse  count: ', worse_cnt)

        # plt.plot(self.rst[self.cfg['plots'][1]['label']]['data'],  # MFD
        #          self.rst[self.cfg['plots'][0]['label']]['data'],  # GFD or EECBS
        #          color='dodgerblue',
        #          marker='.',
        #          zorder=3,
        #          alpha=self.cfg['alpha'],
        #          linewidth=self.cfg['line_width'],
        #          ms=self.cfg['marker_size'])

        ###### Runtime comparison #####

        better_cnt = 0
        worse_cnt = 0
        valid_cnt = 0
        for idx, val1 in enumerate(self.rst[self.cfg['plots'][1]['label']]['data']):
            val0 = self.rst[self.cfg['plots'][0]['label']]['data'][idx]
            # if val0 == self.cfg['time_limit'] or val1 == self.cfg['time_limit']:
            #     invalid_cnt += 1
            #     continue
            valid_cnt += 1
            if val1 < val0:
                better_cnt += 1
            if val1 > val0:
                worse_cnt += 1
        print('valid_cnt: ', valid_cnt)
        print('Better ratio: ', better_cnt / valid_cnt)
        print('Worse  ratio: ', worse_cnt / valid_cnt)

        plt.plot(self.rst[self.cfg['plots'][1]['label']]['data'][:25],
                 self.rst[self.cfg['plots'][0]['label']]['data'][:25],
                 label='city',
                 color='orange',
                 marker='.',
                 zorder=3,
                 alpha=self.cfg['alpha'],
                 linewidth=self.cfg['line_width'],
                 ms=self.cfg['marker_size'])
        plt.plot(self.rst[self.cfg['plots'][1]['label']]['data'][25:50],
                 self.rst[self.cfg['plots'][0]['label']]['data'][25:50],
                 label='den520d',
                 color='purple',
                 marker='.',
                 zorder=2,
                 alpha=self.cfg['alpha'],
                 linewidth=self.cfg['line_width'],
                 ms=self.cfg['marker_size'])
        plt.plot(self.rst[self.cfg['plots'][1]['label']]['data'][50:75],
                 self.rst[self.cfg['plots'][0]['label']]['data'][50:75],
                 label='ost003d',
                 color='teal',
                 marker='.',
                 zorder=1,
                 alpha=self.cfg['alpha'],
                 linewidth=self.cfg['line_width'],
                 ms=self.cfg['marker_size'])
        plt.plot(self.rst[self.cfg['plots'][1]['label']]['data'][75:100],
                 self.rst[self.cfg['plots'][0]['label']]['data'][75:100],
                 label='warehouse',
                 color='dodgerblue',
                 marker='.',
                 zorder=0,
                 alpha=self.cfg['alpha'],
                 linewidth=self.cfg['line_width'],
                 ms=self.cfg['marker_size'])

        plt.xscale('log')
        # ax=plt.gca()
        # ax.xaxis.set_minor_formatter(ScalarFormatter())
        # plt.xticks(fontsize=self.cfg['text_size']['x_axis'])
        # plt.xticks([r*self.cfg['x_axis']['scale'] for r in self.cfg['x_axis']['range']],
        #            labels=self.cfg['x_axis']['range'],
        #            fontsize=self.cfg['text_size']['x_axis'])
        plt.xlabel(self.cfg['x_axis']['label'],
                   fontsize=self.cfg['text_size']['x_axis'])

        plt.yscale('log')
        # ax.yaxis.set_minor_formatter(ScalarFormatter())
        # plt.yticks(fontsize=self.cfg['text_size']['y_axis'])
        # plt.yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
        #            labels=self.cfg['y_axis']['range'],
        #            fontsize=self.cfg['text_size']['y_axis'])
        plt.ylabel(self.cfg['y_axis']['label'],
                   fontsize=self.cfg['text_size']['y_axis'])

        plt.tight_layout()
        if self.cfg['set_legend']:
            plt.legend(fontsize=self.cfg['text_size']['title'])
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input')
    parser.add_argument('--config', type=str, default='single_plot.yaml')
    args = parser.parse_args()

    mapf_plotter = MAPFPlotter(args.config)
    # ------ Average results per agent/weights -----
    mapf_plotter.get_val()
    mapf_plotter.plot_fig()
    # ------ Instance-wise results per agent/weights -----
    # mapf_plotter.get_val()
    # mapf_plotter.plot_fig_instance()
    # ------ Scatter comparison between two features in a solver -----
    # mapf_plotter.get_scatter_val()
    # mapf_plotter.plot_scatter()
    # ------ Scatter comparison between two solvers -----
    # mapf_plotter.get_scatter_comparison()
    # mapf_plotter.plot_scatter_comparison()
