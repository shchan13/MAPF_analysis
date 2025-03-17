"""Iteration processor"""

import os
import sys
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import util
import yaml

DESTROY_STRATEGY = {0: 'Random',
                    1: 'Agent-based',
                    2: 'Intersection-based'}
INT_MAX = 2147483647

class IterProcessor:
    def __init__(self, in_config) -> None:
        self.cfg: Dict = {}
        self.rst:Dict = {}  # Each element is a dataframe from the file

        config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  in_config)
        with open(config_dir, encoding='utf-8', mode='r') as fin:
            self.cfg = yaml.load(fin, Loader=yaml.FullLoader)

        if 'start_iter' not in self.cfg:
            self.cfg['start_iter'] = 0
        if 'end_iter' not in self.cfg:
            self.cfg['end_iter'] = np.inf

        # Create a module spec from the given path in configuration
        self.spec = spec_from_file_location('iter_operate', self.cfg['y_axis']['script'])

        self.func = module_from_spec(self.spec)  # Load the module from the spec
        sys.modules['iter_operate'] = self.func  # Add the module to sys.modules

        # Execute the module to make its attributes accessible
        self.spec.loader.exec_module(self.func)

        # Create the folder for self.cfg['output_dir']
        if not os.path.isdir(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])


    def get_iter_val(self):
        for p in self.cfg['plots']:
            self.rst[p['label']] = {'runtime': [], 'data': [], 'expand_from': []}
            df = util.read_file(p['data'])
            end_iter = min(self.cfg['end_iter'], len(df)-1)
            df = df[self.cfg['start_iter']:end_iter+1]
            for _, row in df.iterrows():
                row_val = self.func.y_operate(row, self.cfg)
                self.rst[p['label']]['data'].append(row_val)
                self.rst[p['label']]['runtime'].append(row['runtime'])
                self.rst[p['label']]['expand_from'].append(row['expand_from'])


    def plot_fig(self) -> None:
        print('Plot figure... ', end='')
        plt.close('all')
        plt.figure(figsize=(self.cfg['fig_width'], self.cfg['fig_height']))
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        max_x_len = 0
        for p in self.cfg['plots']:
            max_x_len = max(max_x_len, len(self.rst[p['label']]))

        for p in self.cfg['plots']:
            x_pos = []
            if self.cfg['x_axis']['feature'] == 'iteration':
                x_pos = list(range(1, max_x_len + 1))
            else:
                x_pos = self.rst[p['label']]['runtime']

            y_pos = self.rst[p['label']]['data']
            for (yid, yval) in enumerate(y_pos):
                if yval == INT_MAX:
                    y_pos[yid] = np.inf
                else:
                    y_pos[yid] -= self.cfg['y_axis']['offset']

            plt.plot(x_pos, y_pos,
                     label=p['label'],
                     color=p['color'],
                     marker=p['marker'],
                     zorder=p['zorder'],
                     alpha=self.cfg['alpha'],
                     markerfacecolor=p['markerfacecolor'],
                     linewidth=self.cfg['line_width'],
                     markeredgewidth=self.cfg['marker_width'],
                     ms=self.cfg['marker_size'])

        plt.xticks(list(self.cfg['x_axis']['range']),
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
        if self.cfg['set_legend']:
            plt.legend(loc='best', fontsize=self.cfg['text_size']['legend'])
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']), dpi=200)
        plt.show()
        print('Done!')


    def plot_bar(self) -> None:
        def get_exp_color(idx: int) -> str:
            if idx == 1:  # expand from CLEANUP
                return 'yellowgreen'
            if idx == 2:  # expand from OPEN
                return 'orange'
            if idx == 3: # expand from FOCAL
                return 'dodgerblue'
            return 'white'

        plt.close('all')
        _, ax = plt.subplots(1, figsize=(self.cfg['fig_width'], 1))
        for pid, p in enumerate(self.cfg['plots']):
            x_pos:List = self.rst[p['label']]['runtime']

            # Define rectangle parameters: (x, y) of bottom-left corner, width, height
            w = x_pos[0]
            rect = patches.Rectangle(xy=(0, pid), width=w, height=1, facecolor='grey')
            ax.add_patch(rect)

            for xid in range(len(x_pos) - 1):
                w = x_pos[xid + 1] - x_pos[xid]
                c = get_exp_color(self.rst[p['label']]['data'][xid])
                rect = patches.Rectangle(xy=(x_pos[xid], pid), width=w, height=1, facecolor=c)
                ax.add_patch(rect)

        ax.set_yticks([])
        plt.xticks(list(self.cfg['x_axis']['range']),
                   labels=self.cfg['x_axis']['range'],
                   fontsize=self.cfg['text_size']['x_axis'])
        plt.xlabel(self.cfg['x_axis']['label'],
                   fontsize=self.cfg['text_size']['x_axis'])
        plt.subplots_adjust(left= 0.05, right=0.95, bottom=0.5, top=0.99)
        plt.show()


    def plot_fig_with_bar(self) -> None:
        def get_exp_color(idx: int) -> str:
            if idx == 1:  # expand from CLEANUP
                return 'yellowgreen'
            if idx == 2:  # expand from OPEN
                return 'orange'
            if idx == 3: # expand from FOCAL
                return 'dodgerblue'
            return 'white'

        plt.close('all')
        _, ax = plt.subplots(figsize=(self.cfg['fig_width'] + 0.5, self.cfg['fig_height'] + 0.5),
                             gridspec_kw={'height_ratios': [10, 1]},
                             constrained_layout=True,
                             nrows=2, ncols=1, sharex=True)
        if 'title' in self.cfg.keys():
            ax[0].set_title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        for pid, p in enumerate(self.cfg['plots']):
            x_pos:List = self.rst[p['label']]['runtime']


            y_pos = self.rst[p['label']]['data']
            y_pos.pop()
            for (yid, yval) in enumerate(y_pos):
                if yval == INT_MAX:
                    y_pos[yid] = np.inf
                else:
                    y_pos[yid] -= self.cfg['y_axis']['offset']

            ax[0].plot(x_pos, y_pos,
                       label=p['label'],
                       color=p['color'],
                       marker=p['marker'],
                       zorder=p['zorder'],
                       alpha=self.cfg['alpha'],
                       markerfacecolor=p['markerfacecolor'],
                       linewidth=self.cfg['line_width'],
                       markeredgewidth=self.cfg['marker_width'],
                       ms=self.cfg['marker_size'])

            # Define rectangle parameters: (x, y) of bottom-left corner, width, height
            w = x_pos[0]
            rect = patches.Rectangle(xy=(0, pid), width=w, height=1, facecolor='grey')
            ax[1].add_patch(rect)

            for xid in range(len(x_pos) - 1):
                w = x_pos[xid + 1] - x_pos[xid]
                c = get_exp_color(self.rst[p['label']]['expand_from'][xid])
                rect = patches.Rectangle(xy=(x_pos[xid], pid), width=w, height=1, facecolor=c)
                ax[1].add_patch(rect)

        ax[0].set_yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
                         labels=self.cfg['y_axis']['range'],
                         fontsize=self.cfg['text_size']['y_axis'])
        ax[0].set_ylabel(self.cfg['y_axis']['label'],
                         fontsize=self.cfg['text_size']['y_axis'])
        ax[0].grid(axis='y')
        ax[1].set_yticks([])
        ax[1].set_xticks(list(self.cfg['x_axis']['range']),
                         labels=self.cfg['x_axis']['range'],
                         fontsize=self.cfg['text_size']['x_axis'])
        ax[1].set_xlabel(self.cfg['x_axis']['label'],
                         fontsize=self.cfg['text_size']['x_axis'])

        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']), dpi=200)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    iter_proc = IterProcessor(args.config)
    iter_proc.get_iter_val()
    # iter_proc.plot_fig()
    # iter_proc.test()
    # iter_proc.plot_bar()
    iter_proc.plot_fig_with_bar()
