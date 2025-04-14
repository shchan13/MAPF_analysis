"""Iteration processor"""

import os
import sys
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import util
import yaml


class PathProcessor:
    def __init__(self, in_config) -> None:
        self.cfg: Dict = {}
        self.rst:Dict = {}  # Each element is a dataframe from the file

        config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  in_config)
        with open(config_dir, encoding='utf-8', mode='r') as fin:
            self.cfg = yaml.load(fin, Loader=yaml.FullLoader)

        # Create a module spec from the given path in configuration
        # self.spec = spec_from_file_location('iter_operate', self.cfg['y_axis']['script'])
        # self.func = module_from_spec(self.spec)  # Load the module from the spec
        # sys.modules['iter_operate'] = self.func  # Add the module to sys.modules

        # Execute the module to make its attributes accessible
        # self.spec.loader.exec_module(self.func)

        # Create the folder for self.cfg['output_dir']
        if not os.path.isdir(self.cfg['output_dir']):
            os.makedirs(self.cfg['output_dir'])


    def get_path_val(self):
        for p in self.cfg['plots']:
            self.rst[p['label']] = {}
            df = util.read_file(p['data'])
            for _, row in df.iterrows():
                self.rst[p['label']][row['time_gen']] = {}
                paths_string: List[str] = row['path_data'].split('|')
                paths_string.pop()
                for pth_str in paths_string:
                    pth_str: List[str] = pth_str.split('_')
                    agent_id = int(pth_str[0])
                    cost = int(pth_str[1])
                    lb = int(pth_str[2])
                    conf_num = int(pth_str[3])
                    max_dis_fx = float(pth_str[4])
                    dis_fx = float(pth_str[5])
                    fx_usg = cost - self.cfg['subopt'] * lb
                    fx_usg_ratio = 0
                    if max_dis_fx > 0:
                        fx_usg_ratio = fx_usg / max_dis_fx
                    self.rst[p['label']][row['time_gen']][agent_id] = {
                        'cost': cost,
                        'lb': lb,
                        'conf_num': conf_num,
                        'max_dis_fx': max_dis_fx,
                        'dis_fx': dis_fx,
                        'fx_usg': fx_usg,
                        'fx_usg_ratio': fx_usg_ratio
                    }


    def plot_histogram(self) -> None:
        print('Plot histogram... ', end='')
        plt.close('all')
        matplotlib.rcParams['xtick.labelsize'] = self.cfg['text_size']['x_axis']
        matplotlib.rcParams['ytick.labelsize'] = self.cfg['text_size']['y_axis']

        plot_num = len(self.cfg['plots'])
        fig, axs = plt.subplots(nrows=1, ncols=plot_num,
                                figsize=(self.cfg['fig_width']*plot_num, self.cfg['fig_height']),
                                sharex=True,
                                sharey=True)
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        all_data: Dict[str, List[float]] = {}
        for pid, p in enumerate(self.cfg['plots']):
            _data:List[float] = []
            for time_gen, paths in self.rst[p['label']].items():
                # if time_gen == 0:
                #     continue
                for _, pth in paths.items():
                    if self.cfg['x_axis']['feature'] in ['fx_usg_ratio', 'fx_usg']:
                        if pth[self.cfg['x_axis']['feature']] > 0:
                            _data.append(pth[self.cfg['x_axis']['feature']])
                    else:
                        _data.append(pth[self.cfg['x_axis']['feature']])

            all_data[p['label']] = _data

        x_min = np.inf
        x_max = -np.inf
        for p in self.cfg['plots']:
            if x_min > min(all_data[p['label']]):
                x_min = min(all_data[p['label']])
            if x_max < max(all_data[p['label']]):
                x_max = max(all_data[p['label']])
        x_rng = np.linspace(x_min, x_max, num=self.cfg['bins']+1)

        for pid, p in enumerate(self.cfg['plots']):
            ax = axs[pid] if plot_num > 1 else axs
            ax.hist(all_data[p['label']],
                    label=p['label'],
                    bins=x_rng,
                    color='dodgerblue',
                    weights=np.ones(len(all_data[p['label']]))/len(all_data[p['label']]),
                    density=False)
            if self.cfg['set_legend']:
                ax.legend(loc='best', fontsize=self.cfg['text_size']['legend'])

            ax.set_xlabel(self.cfg['x_axis']['label'], fontsize=self.cfg['text_size']['x_axis'])
            if pid == 0:
                ax.set_ylabel(self.cfg['y_axis']['label'], fontsize=self.cfg['text_size']['y_axis'])
            # ax.set_yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
            #               labels=self.cfg['y_axis']['range'],
            #               fontsize=self.cfg['text_size']['y_axis'])

        fig.tight_layout()
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']), dpi=200)
        plt.show()
        print('Done!')


    def plot_agents(self) -> None:
        print('Plot agents... ', end='')
        plt.close('all')
        plt.close('all')
        matplotlib.rcParams['xtick.labelsize'] = self.cfg['text_size']['x_axis']
        matplotlib.rcParams['ytick.labelsize'] = self.cfg['text_size']['y_axis']

        plot_num = len(self.cfg['plots'])
        fig, axs = plt.subplots(nrows=1, ncols=plot_num,
                                figsize=(self.cfg['fig_width']*plot_num, self.cfg['fig_height']),
                                sharex=True,
                                sharey=True)
        if 'title' in self.cfg.keys():
            plt.title(self.cfg['title'], fontsize=self.cfg['text_size']['title'])

        all_data: Dict[str, List[float]] = {}
        for p in self.cfg['plots']:
            replan_cnt = 0
            _data = [0 for ag in range(self.cfg['agent_num'] + 1)]
            for paths in self.rst[p['label']].values():
                for agent_idx in paths:
                    _data[agent_idx + 1] += 1
                    replan_cnt += 1
            all_data[p['label']] = [d / replan_cnt for d in _data]

        for pid, p in enumerate(self.cfg['plots']):
            ax = axs[pid] if plot_num > 1 else axs
            ax.title.set_text(p['label'])
            ax.bar(x=list(range(1, self.cfg['agent_num']+1)),
                   height=all_data[p['label']][1:])
            ax.set_xticks(self.cfg['x_axis']['range'],
                          labels=self.cfg['x_axis']['range'],
                          fontsize=self.cfg['text_size']['x_axis'])
            ax.set_xlabel(self.cfg['x_axis']['label'], fontsize=self.cfg['text_size']['x_axis'])
            if pid == 0:
                ax.set_ylabel(self.cfg['y_axis']['label'], fontsize=self.cfg['text_size']['y_axis'])
            ax.set_yticks([r*self.cfg['y_axis']['scale'] for r in self.cfg['y_axis']['range']],
                          labels=self.cfg['y_axis']['range'],
                          fontsize=self.cfg['text_size']['y_axis'])

        fig.tight_layout()
        plt.savefig(os.path.join(self.cfg['output_dir'], self.cfg['output_file']), dpi=200)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    path_proc = PathProcessor(args.config)
    path_proc.get_path_val()
    path_proc.plot_histogram()
    # path_proc.plot_agents()
