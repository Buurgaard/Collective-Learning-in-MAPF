import logging
from typing import Callable as Func, List, Tuple

import torch

from ..map import Map
from ..constants import *
from ..agent import agent_placement

"""Reward signal values"""
R_STEP = R.STEP
R_GOAL = R.GOAL
R_COLLISION = R.COLLISION
R_OBSTACLE = R.OBSTACLE
R_WAIT = R.WAIT

COM_RANGE = Com.RANGE


class Swarm(object):
    """ This a wrapper for the all agents """

    def __init__(self, agent_generators: List[Tuple[int, Func]], placement_func: Func, my_map: Map):
        """ Create the swarm """
        self._agents = []
        self._positions = {}

        self.create_swarm(agent_generators, my_map, placement_func)
        self._my_map = my_map
        self._delta_y = torch.zeros(my_map.size_x, my_map.size_y)
        self._delta_x = torch.zeros(my_map.size_x, my_map.size_y)

    def set_out_path(self, path):
        for agent in self._agents:
            agent.set_output_path(path)
    def create_swarm(self, agent_generators: List[Tuple[int, Func]], my_map: Map, placement_func: Func) -> None:
        """ Create the swarm according to the generators

        :agentgenerators: a list of Tuples of [number of agents, generator function]

        """

        self._agents.clear()
        total_number_of_agents = sum([agent_type[0] for agent_type in agent_generators])

        for number, gen in agent_generators:
            for i in range(number):
                position = placement_func(i, total_number_of_agents, my_map)
                agent = gen(position)
                my_map.add_agent_to_map(agent)
                self._agents.append(agent)

    def move_all(self) -> None:
        """ Move all agents in the swarm

        :world: The world
        :returns: None

        """

        """ Changed to make moving the agents parallel """

        move_requests = [agent.move(self._my_map) for agent in self._agents]
        # print("Requested movement from agents")
        # for agent, pos in zip(self._agents, move_requests):
        # print(agent.position, pos)

        rewards = [R_WAIT] * len(move_requests)

        request_map = self._my_map.obstacle()  # obstacles are added to request map
        """ Check if Agent is trying to move out of the map """
        for i in range(len(move_requests)):
            if move_requests[i][0] < 0 or move_requests[i][0] >= self._my_map.size_x or move_requests[i][1] < 0 or \
                    move_requests[i][1] >= self._my_map.size_y:
                move_requests[i] = self._agents[i].position
                rewards[i] = R_OBSTACLE                                                                                 # negative reward for trying to move out of the arena - 4

            request_map[move_requests[i]] += 1                                                                          # keep track of what positions are requested

        """ Check for collisions in requested moves """
        collision = True  # we assume there is collisions

        while collision:
            collision = False
            collision_map = torch.zeros(self._my_map.size_xy)

            for i in range(len(move_requests)):

                if request_map[move_requests[i]] > 1.5:                                                                 # is there any request for the same node/new-position

                    collision = True                                                                                    # Collision detected

                    if self._agents[i].position != move_requests[i]:                                                    # Check if agent is already got request denied

                        rewards[i] = R_COLLISION                                                                        # reward for collisions with other agents
                        if self._my_map.is_free_from_obs(move_requests[i]) is False:
                            rewards[i] = R_OBSTACLE                                                                     # reward for collisions with obstacle

                        collision_map[move_requests[i]] -= 1                                                            # decrement the requested position
                        collision_map[self._agents[i].position] += 1                                                    # move agent back to old position to check for chain collisions
                        move_requests[i] = self._agents[i].position                                                     # change the request
                    else:
                        pass

            request_map += collision_map

        self._delta_y = torch.zeros(self._my_map.size_x, self._my_map.size_y)
        self._delta_x = torch.zeros(self._my_map.size_x, self._my_map.size_y)
        all_moved = False
        any_moved = True
        n = len(self._agents)
        while not all_moved and any_moved:
            # print("Try move:")
            all_moved = True
            any_moved = False
            i = -1
            for agent, pos in zip(self._agents, move_requests):
                i += 1
                # print(agent.position, pos)
                if agent.position == pos:
                    continue
                try:
                    dir_x, dir_y = agent.direction()
                    self._delta_x[agent.position] = dir_x
                    self._delta_y[agent.position] = dir_y
                    self._my_map.move_agent(agent, pos)
                    rewards[i] = R_STEP
                    any_moved = True
                except:
                    # print("Move failed")
                    all_moved = False                                                                                   # stop trying to move agents if non is able to
                    if not any_moved:
                        for agent_j, pos_j in zip(self._agents, move_requests):
                            if agent_j.position == pos and pos_j == agent.position:                                     # Check for edge collisions / swap moves
                                rewards[i] = R_COLLISION                                                                # reward for colliding with agents try to swap position

        for i in range(len(self._agents)):
            if self._agents[i].position == self._agents[i]._goal:  # has the agent reached the goal
                #print("100 points")
                rewards[i] = R_GOAL


        """ rewards are delivered to the agents """
        for i in range(len(self._agents)):
            self._agents[i].set_reward(rewards[i])
            self._agents[i].set_delta_map(self._delta_x, self._delta_y, self._my_map.size_x,  self._my_map.size_y)
            #if i == 0:
            #    self._agents[i].set_delta_map(self._delta_x, self._delta_y, self._my_map.size_x, self._my_map.size_y, pri=True)

    def optimize_agent_models(self):
        #print("Optimize")
        for agent in self._agents:
            agent.optimize_model_step(self._my_map)

    def communication(self):

        for agent in self._agents:
            agent_mem = agent.get_erm_send()
            if agent_mem.full() or (agent_mem.get_ready() and Com.GOAL):
                for receiver in self._agents:
                    if self.inf_norm(agent.position, receiver.position) <= COM_RANGE:
                        receiver.receive(agent_mem)
                agent.clear_erm()
                agent.clear_ready()


    @staticmethod
    def inf_norm(s: Tuple[int, int], r: Tuple[int, int]):
        return max(abs(s[0]-r[0]), abs(s[1]-r[1]))

    def set_positions(self, position: int) -> None:
        """ Set the same position for all agents in swarm

        :position: The node id

        """
        for agent in self._agents:
            agent.position = position

        self._positions.clear()
        self._positions[str(position)] = len(self._agents)

    # def set_agent_id(self) -> None:
    #     logging.info("Agent ID set")
    #     id_nr = 0
    #     for agent in self._agents:
    #         agent._agent_id = id_nr
    #         id_nr += 1
