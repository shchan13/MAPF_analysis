# -*- coding: UTF-8 -*-
""" Plot map """

import os
import logging
import argparse
from typing import List, Tuple, Dict
from enum import Enum
from tkinter import Tk, Canvas, Label, mainloop
import time
import yaml

COLORS: List[str] = ['deepskyblue', 'royalblue', 'orange', 'peru', 'pink',
                     'yellow', 'green', 'violet', 'tomato', 'yellowgreen',
                     'cyan', 'brown', 'olive', 'gray', 'crimson']

class Action(Enum):
    """
    Actions for each agent

    Args:
        Enum (int): 5 actions including going up, right, down, left, and wait
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    WAIT = 4

class MAPFRenderer:
    """
    Render MAPF instance
    """
    def __init__(self, in_config) -> None:
        self.config: Dict = {}
        config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), in_config)
        with open(config_dir, mode='r', encoding='utf-8') as fin:
            self.config = yaml.load(fin, Loader=yaml.FullLoader)

        self.width: int = -1
        self.height: int = -1
        self.env_map: List[List[bool]] = []

        self.num_of_agents: int = self.config['num_of_agents']
        self.agents: List = []
        self.agent_texts: List = []
        self.start_loc: Dict[int,Tuple[int, int]] = {}
        self.goal_loc: Dict[int,Tuple[int,int]] = {}
        self.paths: Dict[int,List[Tuple[int,int]]] = {}
        self.cur_loc: Dict[int,Tuple[int,int]] = {}
        self.cur_timestep: int = 0
        self.makespan = -1
        self.tile_size = self.config['pixel_per_move'] * self.config['moves']

        self.load_map()
        # self.load_paths()
        self.load_agents()

        self.window = Tk()
        wd_width = str(self.width * self.tile_size + 10)
        wd_height = str(self.height * self.tile_size + 60)
        self.window.geometry(wd_width + 'x' + wd_height)
        self.window.title('MAPF Instance')

        self.timestep_label = Label(self.window,
                              text = f'Timestep: {self.cur_timestep:03d}',
                              font=('Arial', int(self.tile_size)))
        self.timestep_label.pack(side='top', anchor='ne')

        self.canvas = Canvas(width=self.width * self.tile_size,
                             height=self.height * self.tile_size,
                             bg='white')
        self.canvas.pack(side='bottom', pady=5)
        self.render_env()
        self.render_static_positions(loc=self.start_loc, color_idx=0, shape='oval')
        self.render_static_positions(loc=self.goal_loc, color_idx=2, shape='oval')
        # self.render_positions()
        self.canvas.update()
        time.sleep(self.config['delay']*4)


    def render_static_positions(self, loc:List=None, color_idx:int=2, shape:str='rec') -> None:
        if loc is None:
            loc = self.cur_loc
        for _ag_ in range(self.num_of_agents):
            if shape == 'rec':
                self.canvas.create_rectangle(loc[_ag_][0] * self.tile_size,
                                             loc[_ag_][1] * self.tile_size,
                                             (loc[_ag_][0]+1) * self.tile_size,
                                             (loc[_ag_][1]+1) * self.tile_size,
                                             fill=COLORS[color_idx],
                                             outline='')
            elif shape == 'oval':
                self.canvas.create_oval(loc[_ag_][0] * self.tile_size,
                                        loc[_ag_][1] * self.tile_size,
                                        (loc[_ag_][0]+1) * self.tile_size,
                                        (loc[_ag_][1]+1) * self.tile_size,
                                        fill=COLORS[color_idx],
                                        outline='')

            if self.config['plot_ag_num']:
                self.canvas.create_text((loc[_ag_][0]+0.5)*self.tile_size,
                                        (loc[_ag_][1]+0.5)*self.tile_size,
                                        text=str(_ag_+1),
                                        fill='black',
                                        font=('Arial', int(self.tile_size*0.5)))


    def render_env(self):
        for rid, _cur_row_ in enumerate(self.env_map):
            for cid, _cur_ele_ in enumerate(_cur_row_):
                if _cur_ele_ is False:
                    self.canvas.create_rectangle(cid * self.tile_size,
                                                 rid * self.tile_size,
                                                 (cid+1)*self.tile_size,
                                                 (rid+1)*self.tile_size,
                                                 fill='black')


    def render_positions(self, loc: List = None) -> None:
        if loc is None:
            loc = self.cur_loc

        for _ag_ in range(self.num_of_agents):
            color_idx = 0 if _ag_ < 90 else 2
            agent = self.canvas.create_oval(loc[_ag_][0] * self.tile_size,
                                            loc[_ag_][1] * self.tile_size,
                                            (loc[_ag_][0]+1) * self.tile_size,
                                            (loc[_ag_][1]+1) * self.tile_size,
                                            fill=COLORS[color_idx],
                                            outline='')
            self.agents.append(agent)

            if self.config['plot_ag_num']:
                ag_idx = self.canvas.create_text((loc[_ag_][0]+0.5)*self.tile_size,
                                                 (loc[_ag_][1]+0.5)*self.tile_size,
                                                 text=str(_ag_+1),
                                                 fill='black',
                                                 font=('Arial', int(self.tile_size*0.6)))
                self.agent_texts.append(ag_idx)


    def load_map(self, map_file:str = None) -> None:
        if map_file is None:
            map_file = self.config['map_file']

        with open(map_file, mode='r', encoding='utf-8') as fin:
            fin.readline()  # ignore type
            self.height = int(fin.readline().strip().split(' ')[1])
            self.width  = int(fin.readline().strip().split(' ')[1])
            fin.readline()  # ignore 'map' line
            for line in fin.readlines():
                out_line: List[bool] = []
                for word in list(line.strip()):
                    if word == '.':
                        out_line.append(True)
                    else:
                        out_line.append(False)
                assert len(out_line) == self.width
                self.env_map.append(out_line)
        assert len(self.env_map) == self.height

        obstacle_num = 0
        degree_vertex_num = [0, 0, 0, 0, 0]
        for h in range(self.height):
            for w in range(self.width):
                if not self.env_map[h][w]:
                    obstacle_num += 1
                    continue
                cur_degree = 0
                if (w - 1) >= 0 and self.env_map[h][w - 1]:
                    cur_degree += 1
                if (w + 1) <= (self.width - 1) and self.env_map[h][w + 1]:
                    cur_degree += 1
                if (h - 1) >= 0 and self.env_map[h - 1][w]:
                    cur_degree += 1
                if (h + 1) <= (self.height - 1) and self.env_map[h + 1][w]:
                    cur_degree += 1
                degree_vertex_num[cur_degree] += 1
        print('obstacle num:', obstacle_num)
        print('degree_vertex_num: ', degree_vertex_num)

    def load_agents(self, scen_file:str = None) -> None:
        """ load agents' locations from the scen_file

        Args:
            scen_file (str, optional): the location of the scen file. Defaults to None.
        """
        if scen_file is None:
            scen_file = self.config['scen_file']

        with open(scen_file, mode='r', encoding='utf-8') as fin:
            fin.readline()  # ignore the first line 'version 1'
            ag_counter:int = 0
            for line in fin.readlines():
                line_seg = line.split('\t')
                self.start_loc[ag_counter] = (int(line_seg[4]), int(line_seg[5]))
                self.cur_loc[ag_counter] = (int(line_seg[4]), int(line_seg[5]))
                self.goal_loc[ag_counter] = (int(line_seg[6]), int(line_seg[7]))

                ag_counter += 1
                if ag_counter == self.num_of_agents:
                    break


    def load_paths(self, path_file:str = None) -> None:
        """ load paths from the path_file

        Args:
            path_file (str, optional): the location of the path file. Defaults to None.
        """
        if path_file is None:
            path_file = self.config['path_file']
        if not os.path.exists(path_file):
            logging.warning('No path file is found!')
            return

        with open(path_file, mode='r', encoding='utf-8') as fin:
            ag_counter = 0
            for line in fin.readlines():
                ag_idx = int(line.split(' ')[1].split(':')[0])
                self.paths[ag_idx] = []
                for cur_loc in line.split(' ')[-1].split('->'):
                    if cur_loc == '\n':
                        continue
                    cur_x = int(cur_loc.split(',')[1].split(')')[0])
                    cur_y = int(cur_loc.split(',')[0].split('(')[1])
                    self.paths[ag_idx].append((cur_x, cur_y))
                ag_counter += 1
            self.num_of_agents = ag_counter

        for ag_idx in range(self.num_of_agents):
            if self.makespan < len(self.paths[ag_idx]):
                self.makespan = len(self.paths[ag_idx])


    def move_agents(self) -> None:
        """ Move agents from cur_timstep to cur_timestep+1 and increase the cur_timestep by 1
        """

        while self.cur_timestep < self.makespan:
            self.timestep_label.config(text = f'Timestep: {self.cur_timestep:03d}')

            for _ in range(self.config['moves']):
                for ag_idx, agent in enumerate(self.agents):
                    next_timestep = min(self.cur_timestep+1, len(self.paths[ag_idx])-1)
                    direction = (self.paths[ag_idx][next_timestep][0] - self.cur_loc[ag_idx][0],
                                 self.paths[ag_idx][next_timestep][1] - self.cur_loc[ag_idx][1])
                    delta_per_move = (direction[0] * (self.tile_size // self.config['moves']),
                                      direction[1] * (self.tile_size // self.config['moves']))
                    self.canvas.move(agent, delta_per_move[0], delta_per_move[1])
                    self.canvas.move(self.agent_texts[ag_idx], delta_per_move[0], delta_per_move[1])

                self.canvas.update()
                time.sleep(self.config['delay'])

            for ag_idx in range(self.num_of_agents):
                next_timestep = min(self.cur_timestep+1, len(self.paths[ag_idx])-1)
                self.cur_loc[ag_idx] = (self.paths[ag_idx][next_timestep][0],
                                        self.paths[ag_idx][next_timestep][1])
            self.cur_timestep += 1
            time.sleep(self.config['delay'] * 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take config.yaml as input!')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    mapf_renderer = MAPFRenderer(args.config)
    mainloop()
