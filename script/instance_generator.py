"""Generator for one-shot MAPF instance"""

import os
import sys
import argparse
from typing import List, Tuple, Dict
import random
import numpy as np
from script import util

RANDOM_WALK_WEIGHT = 2
GA_NUM_ITERATION  = 10000
MUTATION_PROB = 0.01


def guissian_sampling(mean:float, std:float):
    return max(int(random.gauss(mean, std)), 0)


class Agent:
    def __init__(self, start_loc:Tuple[int,int], goal_loc:Tuple[int,int]):
        self.start_loc:Tuple[int,int] = start_loc
        self.goal_loc:Tuple[int,int]  = goal_loc


class Instance:
    def __init__(self):
        self.agents:List[Agent] = []

    def generate_agents(self, start_loc:List[Tuple[int,int]], goal_loc:List[Tuple[int,int]]):
        assert len(start_loc) == len(goal_loc)
        self.agents = [Agent(start_loc[i], goal_loc[i]) for i in range(len(start_loc))]


class InstanceGenerator:
    """ Generator for MAPF instance
    """

    def __init__(self, map_file:str, num_agents:int, num_ins:int):
        # Initialize the parameters
        self.height, self.width, self.map, self.num_freespace = util.load_map(map_file)
        if num_agents > self.num_freespace:
            print('ERROR: number of agents should be at most the number of free spaces!')
            sys.exit()

        self.map_file = map_file
        self.num_of_agents = num_agents
        self.num_of_ins = num_ins
        self.map_name = util.get_map_name(map_file)
        self.instances:List[Instance] = []
        self.num_of_steps:int = self.num_freespace * RANDOM_WALK_WEIGHT
        self.is_lcc = False


    def valid_loc(self, loc:Tuple[int,int]):
        return 0 <= loc[0] < self.height and 0 <= loc[1] < self.width \
            and self.map[loc[0]][loc[1]]


    def find_lcc(self):  # Find the largest connected component
        print('Find the largest connected component...', end='')
        ccm_idx = 0
        ccm_cnt:Dict = {ccm_idx: 0}
        ccm = [[-1 for _ in range(self.width)] for _ in range(self.height)]

        # Filter out the obstacles
        for ii in range(self.height):
            for jj in range(self.width):
                if self.map[ii][jj] is False:
                    ccm[ii][jj] = -2

        for row_ in range(self.height):
            for col_ in range(self.width):
                if ccm[row_][col_] == -1:
                    start_loc:Tuple[int,int] = (row_, col_)
                    open_list:List[Tuple[int,int]] = [start_loc]
                    while len(open_list) > 0:  # if open list is not empty
                        curr:Tuple[int,int] = open_list.pop(0)
                        if ccm[curr[0]][curr[1]] > -1:
                            continue
                        ccm[curr[0]][curr[1]] = ccm_idx
                        ccm_cnt[ccm_idx] += 1
                        next_loc = [(curr[0]-1, curr[1]), (curr[0]+1, curr[1]),
                                    (curr[0], curr[1]-1), (curr[0], curr[1]+1)]
                        for n_loc in next_loc:
                            if self.valid_loc(n_loc) and ccm[n_loc[0]][n_loc[1]] == -1 \
                                and n_loc not in open_list:
                                open_list.append(n_loc)
                    ccm_idx += 1
                    ccm_cnt[ccm_idx] = 0

        # ccm_arr = np.array(ccm)
        # plt.imshow(ccm_arr, interpolation='none')
        # plt.show()

        if len(ccm_cnt) == 1:
            print('Done!')
            self.is_lcc = True
            return

        ccm_idx = max(zip(ccm_cnt.values(), ccm_cnt.keys()))[1]
        for row_ in range(self.height):
            for col_ in range(self.width):
                if self.map[row_][col_] is True and ccm[row_][col_] != ccm_idx:
                    self.map[row_][col_] = False
        self.is_lcc = True
        print('Done!')


    def generate_default_instances(self):
        """ Generate all instances by default
        """
        for idx in range(self.num_of_ins):
            print('Generate instance '+ str(idx) + '... ')
            self.instances.append(self.generate_instance_by_random_walk())


    def generate_instance_by_lcc(self, num_agents:int=None):
        """ Generate an instance only in the largest connected component
        """
        if self.is_lcc is False:
            self.find_lcc()

        if num_agents is None:
            num_agents = self.num_of_agents
        start_locs = []
        goal_locs  = []
        k = 0
        while k < num_agents:
            # Randomly generate start locations
            srow = random.randint(0, self.height-1)
            scol = random.randint(0, self.width-1)
            if self.map[srow][scol] is False or (srow,scol) in start_locs:
                continue
            if self.map_name == 'warehouse-10-20-10-2-1' and scol in range(25, 136):
                continue
            if self.map_name == 'warehouse-random-64-64-20' and scol in range(31, 97):
                continue
            start_locs.append((srow, scol))
            k += 1

        k = 0
        while k < num_agents:
            # Randomly generate goal locations
            grow = random.randint(0, self.height-1)
            gcol = random.randint(0, self.width-1)
            if self.map[grow][gcol] is False or (grow,gcol) in goal_locs:
                continue
            if self.map_name == 'warehouse-10-20-10-2-1':
                if gcol in range(25, 136)\
                    or (gcol in range(0, 25) and start_locs[k][1] in range(0, 25))\
                    or (gcol in range(136, 161) and start_locs[k][1] in range(136, 161)):  # s2s
                    continue
                # if gcol not in range(25, 136):
                #     continue
            if self.map_name == 'warehouse-random-64-64-20':
                if gcol in range(31, 97)\
                    or (gcol in range(0, 31) and start_locs[k][1] in range(0, 31))\
                    or (gcol in range(97, 128) and start_locs[k][1] in range(97, 128)):  # s2s
                    continue
            goal_locs.append((grow, gcol))
            k += 1

        ins = Instance()
        ins.generate_agents(start_locs, goal_locs)
        return ins


    def generate_instance_by_random_walk(self, num_agents:int=None):
        """ Generate an instance by random walk
        """
        if num_agents is None:
            num_agents = self.num_of_agents

        start_locs = []
        goal_locs  = []
        k = 0
        while k < num_agents:
            # Randomly generate start locations
            srow = random.randint(0, self.height-1)
            scol = random.randint(0, self.width-1)
            if self.map[srow][scol] is False or (srow,scol) in start_locs:
                continue
            start_locs.append((srow, scol))

            # Generate goal locations with random walk
            walk_step = guissian_sampling(self.num_of_steps, self.num_of_steps//4)
            goal_loc = util.random_walk(self.map, (srow, scol), walk_step)
            while goal_loc in goal_locs:
                walk_step = guissian_sampling(self.num_of_steps, self.num_of_steps//4)
                goal_loc = util.random_walk(self.map, (srow, scol), walk_step)
            goal_locs.append(goal_loc)
            print('\tagent ' + str(k), end=' ')
            print('start ', (srow, scol), 'and goal', goal_loc, 'with walk steps ', walk_step)
            k += 1

        ins = Instance()
        ins.generate_agents(start_locs, goal_locs)
        return ins


    def cross_over(self):
        """Crossover for all the instances
        """
        random.shuffle(self.instances)
        for i in range(self.num_of_ins//2):
            j = i + self.num_of_ins//2  # the index of the other instance to cross over

            # Swap the start and goal locations of the same agent index in two instances
            target_ag = np.random.choice(np.arange(self.num_of_agents),
                                         size = self.num_of_agents//2)
            for _ag_ in target_ag:
                self.instances[i].agents[_ag_], self.instances[j].agents[_ag_] =\
                    self.instances[j].agents[_ag_], self.instances[i].agents[_ag_]


    def mutation(self):
        """Randomly pick an instance and mutate
        """
        idx = random.randint(0, self.num_of_ins-1)
        num_new_agents = random.randint(1, max(1, self.num_of_agents//10))
        new_agents = np.random.choice(np.arange(self.num_of_agents), size = num_new_agents)

        if random.random() > 0.8:  # Replace agents with new agents
            tmp_ins = self.generate_instance_by_random_walk(num_new_agents)

            for k in range(num_new_agents):
                is_overlap = False
                for agent in self.instances[idx]:
                    if agent.start_loc == tmp_ins.agents[k].start_loc or\
                        agent.goal_loc == tmp_ins.agents[k].goal_loc:
                        is_overlap = True
                        break
                if not is_overlap:
                    self.instances[idx].agents[new_agents[k]] = tmp_ins.agents[k]

        else:  # Swap the start and goal locations
            for new_ag in new_agents:
                tmp_loc:Tuple[int,int] = self.instances[idx].agents[new_ag].start_loc
                self.instances[idx].agents[new_ag].start_loc =\
                    self.instances[idx].agents[new_ag].goal_loc
                self.instances[idx].agents[new_ag].goal_loc = tmp_loc


    def genetic_algorithm(self):
        """Overall genetic algorithm for optimizing instances
        """
        assert len(self.instances) > 0

        for _ in range(GA_NUM_ITERATION):
            # Add fitness function here
            # End: Add fitness function
            self.cross_over()
            if random.random() < MUTATION_PROB:  # Mutation happens
                self.mutation()


    def write_all_instsances(self, out_directory:str='../local/', label='tmp'):
        """Write the instance to out_directory

        Args:
            out_directory (str, optional): path to the directory. Defaults to '../local/'.
            label (str, optional): what label of the scen file. Defaults to 'tmp'.
        """
        out_directory = os.path.join(out_directory, 'scen-'+label)
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)

        print('Write instances to files...', end='\t')
        for idx, ins in enumerate(self.instances):
            file_name = self.map_name + '-' + label + '-' + str(idx+1) + '.scen'
            write_to = os.path.join(out_directory, file_name)  # Generate path to write
            with open(write_to, mode='w', encoding='utf-8') as fout:
                fout.write('version 1\n')
                for ag_idx, agent in enumerate(ins.agents):
                    wr_ln  = str(ag_idx) + '\t' + self.map_file + '\t'
                    wr_ln += str(self.width) + '\t' + str(self.height) + '\t'
                    wr_ln += str(agent.start_loc[1]) + '\t' + str(agent.start_loc[0]) + '\t'
                    wr_ln += str(agent.goal_loc[1])  + '\t' + str(agent.goal_loc[0])  + '\t'
                    wr_ln += str(0) + '\n'
                    fout.write(wr_ln)
        print('Done!')


    def write_instance(self, instance:Instance, out_directory:str='../local/', label='tmp'):
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        print('Write the current instance to files...', end='\t')
        file_name = self.map_name + '-' + label + '.scen'
        write_to = os.path.join(out_directory, file_name)  # Generate path to write
        with open(write_to, mode='w', encoding='utf-8') as fout:
            fout.write('version 1\n')
            for ag_idx, agent in enumerate(instance.agents):
                wr_ln  = str(ag_idx) + '\t' + self.map_file + '\t'
                wr_ln += str(self.width) + '\t' + str(self.height) + '\t'
                wr_ln += str(agent.start_loc[1]) + '\t' + str(agent.start_loc[0]) + '\t'  # col,row
                wr_ln += str(agent.goal_loc[1])  + '\t' + str(agent.goal_loc[0])  + '\t'  # col,row
                wr_ln += str(0) + '\n'
                fout.write(wr_ln)
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for instacne generator')
    parser.add_argument('--mapFile',  type=str, default='./example/random-32-32-20.map')
    parser.add_argument('--agentNum',  type=int, default=1)
    parser.add_argument('--insNum',  type=int, default=1)
    parser.add_argument('--outDir',  type=str, default='../local')
    parser.add_argument('--label',  type=str, default='tmp')
    parser.add_argument('--startID',  type=int, default=1)
    args = parser.parse_args()

    scen_dir = args.outDir+'/scen-'+args.label
    ins_gen = InstanceGenerator(args.mapFile, args.agentNum, args.insNum)
    for ins_idx in range(args.insNum):
        print('Generate instance '+ str(args.startID + ins_idx) + '... ')
        cur_ins = ins_gen.generate_instance_by_lcc(args.agentNum)
        ins_gen.write_instance(cur_ins, scen_dir, args.label+'-' + str(args.startID + ins_idx))
