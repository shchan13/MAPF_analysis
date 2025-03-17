"""Iteration processor"""

import os
import argparse
from typing import Dict, List
import tqdm
import yaml
import matplotlib as mlb
import matplotlib.pyplot as plt
import util
import numpy as np

DESTROY_STRATEGY = {0: 'Random',
                    1: 'Agent-based',
                    2: 'Intersection-based'}
INT_MAX = 2147483647

class LNSProcessor:
    def __init__(self, in_config) -> None:
        self.cfg: Dict = {}
        config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  in_config)
        with open(config_dir, encoding='utf-8', mode='r') as fin:
            self.cfg = yaml.load(fin, Loader=yaml.FullLoader)

        if 'start_iter' not in self.cfg:
            self.cfg['start_iter'] = 0
        if 'end_iter' not in self.cfg:
            self.cfg['end_iter'] = np.inf

        self.files:str = self.cfg['files']  # [{path, label, color,...}, ...]
        self.x_labels:str = self.cfg['x_labels']
        self.y_labels:str = self.cfg['y_labels']
        self.results:List = []  # Each element is a dataframe from the files


    def get_iter_val(self):
        for curr in self.cfg['files']:
            data_frame = util.read_file(curr['path'])
            end_iter = min(self.cfg['end_iter'], len(data_frame))
            curr['data'] = data_frame[self.cfg['start_iter']:end_iter+1]
            self.results.append(curr)
        if self.cfg['y_labels'] in ['agents', 'destroy weight', 'destroy probability']:
            self.proc_multi_val(self.cfg['y_labels'])


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
                              figsize=(self.cfg['figure_width'],
                                       self.cfg['figure_height']),
                              dpi=80, facecolor='w', edgecolor='k')
        max_x_len = 0
        for rst in self.results:
            max_x_len = max(max_x_len, len(rst['data']))

        left_bd  = -1 * self.cfg['set_shift']
        right_bd = self.cfg['set_shift']
        plt_rng  = (right_bd - left_bd) / len(self.results)

        for rid,rst in enumerate(self.results):
            mf_color= 'white' if 'markerfacecolor' not in rst.keys() else rst['markerfacecolor']
            zord = 0 if 'zorder' not in rst.keys() else rst['zorder']

            x_pos = []
            if self.cfg['x_labels'] == 'iteration':
                x_pos = [x + plt_rng * rid for x in range(1, max_x_len+1)]
            else:
                x_pos = rst['data'][self.cfg['x_labels']].to_list()

            y_pos = rst['data'][self.cfg['y_labels']].to_list()

            if self.cfg['y_labels'] == 'sum of costs':
                lb = rst['data']['sum of distances'].to_list()[0]
                y_pos = [y - lb for y in y_pos]

                axs.plot(x_pos, y_pos,
                         label=rst['label'],
                         linewidth=self.cfg['line_width'],
                         marker=rst['marker'],
                         ms=self.cfg['marker_size'],
                         markerfacecolor=mf_color,
                         markeredgewidth=self.cfg['marker_width'],
                         color=rst['color'],
                         alpha=self.cfg['alpha'],
                         zorder=zord)

            elif self.cfg['y_labels'] in ['destroy probability', 'destroy weight']:
                y_pos.pop(0)
                y_pos.insert(0,[1/3., 1/3., 1/3.])
                x_pos.insert(0, 0.0)
                widths = [x_pos[ii] - x_pos[ii-1] for ii in range(1,len(x_pos))]
                x_pos.pop(-1)
                bottom = [0 for _ in x_pos]
                for idx in range(len(y_pos[0])):  # idx: indices of destroy operators
                    cur_y:List = []
                    for yvals in y_pos:
                        if yvals[idx] > INT_MAX:
                            yvals[idx] = np.inf
                        cur_y.append(yvals[idx])
                    axs.bar(x_pos, cur_y, width=widths, align='edge', bottom=bottom,
                            label=DESTROY_STRATEGY[idx])
                    bottom = [bottom[j] + cur_y[j] for j in range(len(bottom))]

            elif self.cfg['y_labels'] == 'agents':
                replan_ag = []
                for yvals in y_pos:
                    cur_ag = [False] * self.cfg['agent_num']
                    if yvals[0] != -1.0:
                        for ag_ in yvals:
                            cur_ag[int(ag_)] = True
                    replan_ag.append(cur_ag)
                replan_ag = np.array(replan_ag).T.tolist()

                if self.cfg['x_axis']['feature'] == 'iteration':
                    axs.imshow(replan_ag, aspect='auto',
                               cmap=mlb.colors.ListedColormap(['white', 'black']))
                    plt.xticks(np.arange(self.cfg['start_iter'],
                                        self.cfg['start_iter']+len(replan_ag[0])+1,
                                        len(replan_ag[0])/float(self.cfg['num_x_axis']),
                                        dtype=np.int64))

                elif self.cfg['x_axis']['feature'] == 'agents':
                    x_pos = list(range(1, self.cfg['agent_num']+1))
                    y_pos = [sum(ag_in_iter) for ag_in_iter in replan_ag]
                    axs.bar(x_pos, y_pos, width=1.0,
                            color=rst['color'],
                            label=rst['label'],
                            alpha=self.cfg['alpha'],
                            zorder=zord)

            else:
                for (yid,yval) in enumerate(y_pos):
                    if yval == INT_MAX:
                        y_pos[yid] = np.inf
                axs.plot(x_pos, y_pos,
                         label=rst['label'],
                         linewidth=self.cfg['line_width'],
                         marker=rst['marker'],
                         ms=self.cfg['marker_size'],
                         markerfacecolor=mf_color,
                         markeredgewidth=self.cfg['marker_width'],
                         color=rst['color'],
                         alpha=self.cfg['alpha'],
                         zorder=zord)

        # x_labels = [int(x) for x in x_labels]
        # axs.axes.set_xticklabels(x_labels, fontsize=self.cfg['text_size'])

        # y_labels = axs.axes.get_yticks()
        # y_labels = [int(y) for y in y_labels]
        # axs.axes.set_yticklabels(y_labels, fontsize=self.cfg['text_size'])
        plt.xticks(fontsize=self.cfg['text_size'])
        plt.xlabel(self.cfg['x_labels']['label'].capitalize(),
                   fontsize=self.cfg['text_size'])

        if self.cfg['y_labels'] in ['destroy probability', 'destroy weight']:
            plt.yticks(np.arange(0, 1.01, 0.2))
        plt.yticks(fontsize=self.cfg['text_size'])

        if self.cfg['y_labels'] == 'agents' and self.cfg['x_labels'] == 'agents':
            self.cfg['y_labels'] = 'number of replans'
        plt.ylabel(self.cfg['y_labels']['label'].capitalize(),
                   fontsize=self.cfg['text_size'])

        # if self.cfg['y_labels'] in ['destroy probability', 'destroy weight']:
        #     handles, labels = plt.gca().get_legend_handles_labels()
        #     order = list(DESTROY_STRATEGY.keys())
        #     order.reverse()
        #     plt.legend([handles[i] for i in order], [labels[i] for i in order],
        #                loc="best", fontsize=self.cfg['text_size'])
        # else:
        #     plt.legend(loc="best", fontsize=self.cfg['text_size'])
        plt.tight_layout()
        plt.savefig(self.cfg['save_path'], dpi=200)
        plt.show()
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    iter_proc = LNSProcessor(args.config)
    iter_proc.get_iter_val()
    iter_proc.plot_fig()
