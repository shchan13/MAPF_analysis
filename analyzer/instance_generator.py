"""Generator for MAPF instance
"""
# Instance generator for one-shot MAPF

import os
import argparse
from typing import List, Tuple
import random
import scipy.stats as ss
import numpy as np
import util

RANDOM_WALK_STEPS = 100000
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
        self.map_file = map_file
        self.num_of_agents = num_agents
        self.num_of_ins = num_ins
        self.height, self.width, self.map = util.load_map(map_file)
        self.map_name = util.get_map_name(map_file)
        self.instances:List[Instance] = []


    def generate_default_instances(self):
        """ Generate all instances by default
        """
        for idx in range(self.num_of_ins):
            print('Generate instance '+ str(idx) + '... ',)
            self.instances.append(self.generate_instance())


    def generate_instance(self, num_agents:int=None):
        """ Generate an instance
        """
        if num_agents is None:
            num_agents = self.num_of_agents

        start_locs = []
        goal_locs  = []
        k = 0
        while k < num_agents:
            # Random generate start location
            srow = random.randint(0, self.width-1)
            scol = random.randint(0, self.height-1)
            if self.map[srow][scol] is False or (srow,scol) in start_locs:
                continue
            start_locs.append((srow, scol))

            # Generate goal location with random walk
            walk_step = guissian_sampling(RANDOM_WALK_STEPS, RANDOM_WALK_STEPS//4)
            goal_loc = util.random_walk(self.map, (srow, scol), walk_step)
            while goal_loc in goal_locs:
                walk_step = guissian_sampling(RANDOM_WALK_STEPS, RANDOM_WALK_STEPS//4)
                goal_loc = util.random_walk(self.map, (srow, scol), walk_step)
            goal_locs.append(goal_loc)
            print('\tagent ' + str(k), end=' ')
            print('start ', (srow, scol), 'and goal', goal_loc, 'with walk steps ', walk_step)
            k += 1

        cur_ins = Instance()
        cur_ins.generate_agents(start_locs, goal_locs)
        return cur_ins


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
        ins_idx = random.randint(0, self.num_of_ins-1)
        num_new_agents = random.randint(1, max(1, self.num_of_agents//10))
        new_agents = np.random.choice(np.arange(self.num_of_agents), size = num_new_agents)

        if random.random() > 0.8:  # Replace agents with new agents
            tmp_ins = self.generate_instance(num_new_agents)

            for k in range(num_new_agents):
                is_overlap = False
                for agent in self.instances[ins_idx]:
                    if agent.start_loc == tmp_ins.agents[k].start_loc or\
                        agent.goal_loc == tmp_ins.agents[k].goal_loc:
                        is_overlap = True
                        break
                if not is_overlap:
                    self.instances[ins_idx].agents[new_agents[k]] = tmp_ins.agents[k]

        else:  # Swap the start and goal locations
            for new_ag in new_agents:
                tmp_loc:Tuple[int,int] = self.instances[ins_idx].agents[new_ag].start_loc
                self.instances[ins_idx].agents[new_ag].start_loc =\
                    self.instances[ins_idx].agents[new_ag].goal_loc
                self.instances[ins_idx].agents[new_ag].goal_loc = tmp_loc


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


    def write_instsances(self, out_directory:str='../local/', label='tmp'):
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
                    write_line  = str(ag_idx) + '\t' + self.map_file + '\t'
                    write_line += str(self.width) + '\t' + str(self.height) + '\t'
                    write_line += str(agent.start_loc[1]) + '\t' + str(agent.start_loc[0]) + '\t'
                    write_line += str(agent.goal_loc[1])  + '\t' + str(agent.goal_loc[0])  + '\t'
                    write_line += str(0) + '\n'
                    fout.write(write_line)
        print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for instacne generator')
    parser.add_argument('--mapFile',  type=str, default='./example/random-32-32-20.map')
    parser.add_argument('--agentNum',  type=int, default=1)
    parser.add_argument('--insNum',  type=int, default=1)
    parser.add_argument('--outDir',  type=str, default='../local')
    parser.add_argument('--label',  type=str, default='tmp')
    args = parser.parse_args()

    ins_gen = InstanceGenerator(args.mapFile, args.agentNum, args.insNum)
    ins_gen.generate_default_instances()
    ins_gen.write_instsances(out_directory=args.outDir, label=args.label)
