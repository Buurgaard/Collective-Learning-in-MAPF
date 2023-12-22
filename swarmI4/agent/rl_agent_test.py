""" Reinforcement Learning Agent """

import logging

import random
import math
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

import random as rnd
from typing import Callable, Tuple
from collections import namedtuple, deque

from .agent_interface import AgentInterface
from ..constants import *

BATCH_SIZE = Net.BATCH_SIZE
GAMMA = Net.GAMMA
EPS_START = Net.EPS_START
EPS_END = Net.EPS_END
EPS_DECAY = Net.EPS_DECAY
TAU = Net.TAU
LR = Net.LR
MEM_SIZE = Net.MEM_SIZE

R_STEP = R.STEP
R_GOAL = R.GOAL
R_COLLISION = R.COLLISION
R_OBSTACLE = R.OBSTACLE
R_WAIT = R.WAIT

FOV = Net.FOV
EXTENDED_FOV = Net.EXTENDED_FOV


action_space = [0, 1, 2, 3, 4]  # ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
data_path = "/data/"
model_path = "/models/"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Replay Memory """
Transitions = namedtuple('Transitions', ('state', 'action', 'reward', 'next_state'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], capacity)

    def push(self, *args):
        """save a transition"""
        self.memory.append(Transitions(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    """ DQN model"""
    def __init__(self, n_observations):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # x = F.softmax(x, dim=1)
        return x


class RLagentModel(AgentInterface):
    n_created = -1

    def __init__(self, position: Tuple[int, int]):

        RLagentModel.n_created += 1

        super(RLagentModel, self).__init__(None, position)
        self._goal = position
        self._a_star_path = []
        self._fov_range = FOV
        self._ex_fov_range = EXTENDED_FOV
        self._agent_id = RLagentModel.n_created
        self._steps_done = 0
        self._saved = False
        self._best_loss = 100
        self._dir_x = 0
        self._dir_y = 0
        self._delta_x_map = torch.zeros(FOV * 2 + 1, FOV * 2 + 1)
        self._delta_y_map = torch.zeros(FOV * 2 + 1, FOV * 2 + 1)

        self._current_state = None
        self._current_action = None
        self._reward = 0
        self._new_state = None
        self._man_dist = 0

        self._steps_to_goal = 0
        self._steps_taken = 0
        self._makespan = 0
        self._model_num = 100

        state_length = ((self._fov_range*2+1)**2)*4 + ((self._ex_fov_range*2+1)**2)*2 + 2
        #state_length = ((self._fov_range*2+1)**2) + ((self._ex_fov_range*2+1)**2)*2 + 2
        print(state_length)
        self._policy_net = DQN(state_length).to(device)
        self._policy_net_dict = self._policy_net.state_dict()
        self._target_net = DQN(state_length).to(device)
        self._target_net_dict = self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = ReplayMemory(MEM_SIZE)
        self._input_path = ""
        self._output_path = ""
        self._model_name = ""
        self._filename = ""
        self._writer = None

    def set_output_path(self, path):
        #print(path)
        head_tail = os.path.split(path)
        #print(head_tail)
        head_tail = os.path.split(head_tail[0])
        #print(head_tail)
        num = head_tail[1]

        self._input_path = head_tail[0]
        # print(head_tail[0])
        self._input_path = str(Path(head_tail[0], "train", num))
        self._output_path = str(Path(head_tail[0], "test", num))
        # print(self._input_path)

        self._model_name = self._input_path + model_path + str(self._agent_id) + "_model"
        self._filename = self._output_path + "/" + str(self._agent_id) + "_data.csv"

        print(self._model_name)
        print(self._filename)

        #Path(self._output_path, model_path).mkdir(parents=True, exist_ok=True)
        Path(self._filename).parent.mkdir(parents=True, exist_ok=True)



        if os.path.exists(self._model_name + str(self._model_num)):
            # print(self._model_name + str(self._model_num))
            checkpoint = torch.load(self._model_name + str(self._model_num))
            self._policy_net.load_state_dict(checkpoint['model_state_dict'])
            self._target_net.load_state_dict(checkpoint['target_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self._policy_net.eval()
            self._target_net.eval()

        with open(self._filename, 'w', newline='') as csvfile:
            fieldnames = ['time_step', 'loss', 'reward', 'on_goal', 'makespan', 'model']
            self._writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            self._writer.writeheader()

    def get_goal(self, my_map):
        #print(self._agent_id)
        #print("New GOAL", self._agent_id)
        tries = 0
        goal_set = False
        max_goal_setting_tries = 100
        #logging.info(my_map)
        while not goal_set or tries > max_goal_setting_tries:
            x, y = random.randint(0, my_map.size_x - 1), random.randint(0, my_map.size_y - 1)
            if not my_map.is_free_from_obs((x, y)):
                goal_set = True
                return x, y
            else:
                tries += 1

        if not goal_set:
            assert False, f"Tried to set goal {tries} times, but without luck"

        return x, y

    def set_reward(self, reward):
        self._reward = reward

    def select_action(self, state):
        # print(state)
        if self._steps_done % 100 == 0 and self._steps_done > 0:
            self._model_num += 100
            if os.path.exists(self._model_name+str(self._model_num)):
                print("load model:", self._model_num)
                checkpoint = torch.load(self._model_name+str(self._model_num))
                self._policy_net.load_state_dict(checkpoint['model_state_dict'])
                self._target_net.load_state_dict(checkpoint['target_state_dict'])
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                self._policy_net.eval()
                self._target_net.eval()
            else:
                print("No Model exists", self._model_name+str(self._model_num))

        sample = random.random()
        #eps_threshold = EPS_END + (EPS_START-EPS_END) * math.exp(-1. * self._steps_done / EPS_DECAY)
        #print("eps_thres", eps_threshold)
        self._steps_done += 1
        #if sample > eps_threshold:
        with torch.no_grad():                                                                                       # will return the largest value
            return self._policy_net(state).max(1)[1].view(1, 1)
        #else:
         #   act = torch.tensor([[random.choice(action_space)]], device=device, dtype=torch.long)
            #print("random", act)

            #return act

    def optimize_model(self):
        print("log")
        self.log_data(None, self._reward)



    def generate_state(self, my_map):
        relative_dist = [self._goal[0] - self.position[0], self._goal[1] - self.position[1]]
        relative_dist = torch.tensor(relative_dist)
        # goal = torch.tensor([self._goal[0], self._goal[1]])
        fov_map = self.get_map_fov(self, my_map.get_map_info(), self.position, self._fov_range)
        ex_fov_map = self.get_map_fov(self, my_map.get_map_info(), self.position, self._ex_fov_range)
        obs_map = RLagentModel.create_obs_map(ex_fov_map)
        agent_map = RLagentModel.create_agent_map(fov_map)
        a_star_valid, a_star_map = self.fov_a_star_map(self._fov_range)
        _, a_star_map_ex = self.fov_a_star_map(self._ex_fov_range)

        if not a_star_valid:
            self._a_star_path = my_map.a_star_map(self._position, self._goal)
            a_star_valid, a_star_map = self.fov_a_star_map(self._fov_range)
            a_star_valid, a_star_map_ex = self.fov_a_star_map(self._ex_fov_range)

        """ Combine info maps/list into state array """
        # [goal, agent_map, a_star_map, delta_x , delta_y ,obstacle_map, ex_a_star]

        current_state = torch.cat((relative_dist, torch.flatten(agent_map), torch.flatten(a_star_map),
                                   torch.flatten(self._delta_x_map), torch.flatten(self._delta_y_map),
                                   torch.flatten(obs_map), torch.flatten(a_star_map_ex)), 0)

        #current_state = torch.cat((relative_dist, torch.flatten(a_star_map),
         #                          torch.flatten(obs_map), torch.flatten(a_star_map_ex)), 0)

        # print(len(torch.flatten(obs_map)))
        return torch.unsqueeze(current_state, dim=0).to(device)

    def move(self, my_map):
        """ Generate a request for a move """

        """ Set new goal if the current goal is reached """
        if self._position == self._goal:
            new_goal = self.get_goal(my_map)
            while self._goal == new_goal:
                new_goal = self.get_goal(my_map)
            self._goal = new_goal
            if self._steps_taken != 0:
                self._makespan = self._steps_to_goal/self._steps_taken

                # print("NEW GOAL POS: ", self._goal,
                  #    "steps to goal: ", self._steps_to_goal,
                  #    "steps taken: ", self._steps_taken,
                  #    "Make span: ", self._makespan)

            #else:
                # print("SAME GOAL: ", self._goal)
            self._a_star_path = my_map.a_star_map(self._position, self._goal)
            self._steps_to_goal = len(self._a_star_path) - 1
            self._steps_taken = 0

        ''' State maps and information - network input '''
        self._man_dist = abs(self._goal[0] - self.position[0]) + abs(self._goal[1] - self.position[1])
        self._current_state = self.generate_state(my_map)

        """ Generate position request """
        action = self.select_action(self._current_state)
        self._current_action = torch.reshape(action[0], (1, 1))
        # print(self._current_action)

        x, y = self._position
        self._dir_x = 0
        self._dir_y = 0
        if action == 0:     # UP
            y += 1
            self._dir_y = 1
            #print("UP")
        if action == 1:     # DOWN
            y -= 1
            self._dir_y = -1
            #print("DOWN")
        if action == 2:     # LEFT
            x -= 1
            self._dir_x = -1
            #print("LEFT")
        if action == 3:     # RIGHT
            x += 1
            self._dir_x = 1
            #print("RIGHT")
        if action == 4:     # WAIT
            #print("WAIT")
            x, y = self._position

        self._steps_taken += 1

        return x, y

    def direction(self):
        return self._dir_x, self._dir_y

    def set_delta_map(self, delta_x, delta_y, map_x, map_y, pri=False):
        expanded_x = torch.zeros(map_x+2*FOV, map_y+2*FOV)
        expanded_y = torch.zeros(map_x + 2 * FOV, map_y + 2 * FOV)
        expanded_x[FOV:map_x+FOV, FOV:map_y+FOV] = delta_x
        expanded_y[FOV:map_x+FOV, FOV:map_y+FOV] = delta_y
        x_pos, y_pos = self.position
        x_pos += FOV
        y_pos += FOV
        self._delta_x_map = expanded_x[x_pos - FOV:x_pos + FOV+1, y_pos - FOV:y_pos + FOV+1]
        self._delta_y_map = expanded_y[x_pos - FOV:x_pos + FOV+1, y_pos - FOV:y_pos + FOV+1]
        #if pri:
         #   print("position", self.position)
         #   print(self._delta_x_map.size(), self._delta_y_map.size())

    def optimize_model_step(self, my_map):
        """ State maps and information - network input """
        self._new_state = self.generate_state(my_map)
        #print("_new_state", self._new_state)

        if self._new_state is not None:
            if self._reward == R_STEP:
                man_dist = abs(self._goal[0] - self.position[0]) + abs(self._goal[1] - self.position[1])
                if man_dist >= self._man_dist:
                    self._reward = R_WAIT
                    #print("moved away")
                #else:
                    #print("moved closer")
            self._memory.push(self._current_state, self._current_action, torch.tensor([[self._reward]]), self._new_state)

        #print("log")
        self.log_data(None, self._reward)

        # self.optimize_model()

        #self._target_net_dict = self._target_net.state_dict()
        #self._policy_net_dict = self._policy_net.state_dict()
        #for key in self._policy_net_dict:
        #   self._target_net_dict[key] = self._policy_net_dict[key]*TAU + self._target_net_dict[key]*(1-TAU)            #  merge the two networks
        #   self._target_net.load_state_dict(self._target_net_dict)

    def log_data(self, loss, reward):
        if reward > 50:
            goal = 1
            makespan = self._steps_to_goal/(self._steps_taken+1)
        else:
            goal = 0
            makespan = None
        steps_done = self._steps_done
        with open(self._filename, 'a') as csvfile:
            # print("makespan: ", makespan)
            fieldnames = ['time_step', 'loss', 'reward', 'on_goal', 'makespan', 'model']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'time_step': steps_done, 'loss': loss, 'reward': reward, 'on_goal': goal,
                             'makespan': makespan, 'model': self._model_num})

    @ staticmethod
    def get_map_fov(self, info_map, position, fov_range):
        fov = fov_range
        pos_x = position[0]
        pos_y = position[1]
        fov_image = torch.ones(fov * 2 + 1, fov * 2 + 1)
        info_map_x = info_map.size(0)
        info_map_y = info_map.size(1)

        x_s_fov = y_s_fov = 0
        x_e_fov = fov_image.size(0) - 1
        y_e_fov = fov_image.size(1) - 1

        x_start = pos_x - fov
        y_start = pos_y - fov
        x_end = pos_x + fov
        y_end = pos_y + fov

        # logging.info(info_map)
        # logging.info(info_map_x)
        # logging.info(info_map_y)
        # logging.info("info map crop")
        # logging.info(x_start)
        # logging.info(x_end)
        # logging.info(y_start)
        # logging.info(y_end)
        # logging.info("FOV map")
        # logging.info(x_s_fov)
        # logging.info(x_e_fov)
        # logging.info(y_s_fov)
        # logging.info(y_e_fov)

        if pos_x < fov:
            x_start = 0
            x_s_fov = abs(pos_x - fov)
        if pos_y < fov:
            y_start = 0
            y_s_fov = abs(pos_y - fov)
        if pos_x + fov >= info_map_x:
            x_e_fov -= x_end - info_map_x + 1
            x_end = info_map_x - 1
        if pos_y + fov >= info_map_y:
            y_e_fov -= y_end - info_map_y + 1
            y_end = info_map_y - 1

        # logging.info("info map crop")
        # logging.info(x_start)
        # logging.info(x_end)
        # logging.info(y_start)
        # logging.info(y_end)
        # logging.info("FOV map")
        # logging.info(x_s_fov)
        # logging.info(x_e_fov)
        # logging.info(y_s_fov)
        # logging.info(y_e_fov)

        fov_image[x_s_fov:x_e_fov + 1, y_s_fov:y_e_fov + 1] = info_map[x_start:x_end + 1, y_start:y_end + 1]

        return fov_image

    @staticmethod
    def create_obs_map(fov_map):
        obs_map = torch.clone(fov_map)
        obs_map[obs_map == 2] = 0

        return obs_map

    @staticmethod
    def create_agent_map(fov_map):
        agn_map = torch.clone(fov_map)
        agn_map[agn_map == 1] = 0
        agn_map[agn_map == 2] = 1
        # print("agents")
        # print(agn_map)

        return agn_map

    def fov_a_star_map(self, fov_range):
        fov_a_star = torch.zeros(fov_range * 2 + 1, fov_range * 2 + 1)
        path_elem_in_fov = 0

        for i in range(len(self._a_star_path)):
            point = self._a_star_path[i]
            x = point[0]
            y = point[1]
            if self._position[0] - fov_range <= x <= self._position[0] + fov_range:
                #print("x", x, self._position[0] - fov_range, self._position[0] + fov_range)
                if self._position[1] - fov_range <= y <= self._position[1] + fov_range:
                    #print("y", y, self._position[1] - fov_range, self._position[1] + fov_range)
                    path_elem_in_fov += 1
                    fov_x = (self._position[0] - x - fov_range) * -1
                    fov_y = (self._position[1] - y - fov_range) * -1
                    #print(fov_x, fov_y)
                    fov_a_star[fov_x, fov_y] = path_elem_in_fov

        if path_elem_in_fov > 0:
            #print(fov_a_star)
            return True, fov_a_star / path_elem_in_fov

        return False, fov_a_star

    def set_id(self, id_nr):
        self._agent_id = id_nr


def rl_agent_model_generator() -> Callable:
    """
    Factory function to create agents if the right type

    @return: a generator for IdaTestAgent
    """

    def generator(position: Tuple[int, int]):
        return RLagentModel(position)

    return generator

