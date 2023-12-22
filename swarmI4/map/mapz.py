""" Contains the world description for the swarm """
from typing import Tuple
import networkx as nx
from itertools import islice
# from networkx.drawing.nx_agraph import graphviz_layout
import logging

import torch

from ..agent import AgentInterface

from .color import FREE, AGENT, OBSTACLE


class Map(object):
    """ The world representation """

    def __init__(self, graph, number_of_nodes: Tuple[int, int]):
        """TODO: to be defined. """
        logging.debug("Initializing the world")
        self._graph = graph
        self._number_of_nodes = number_of_nodes

    @property
    def size_xy(self) -> Tuple[int, int]:
        return self._number_of_nodes

    @property
    def size_x(self) -> int:
        return self._number_of_nodes[0]

    @property
    def size_y(self) -> int:
        return self._number_of_nodes[1]

    def connected(self, node: int) -> list:
        """ Get the corrected nodes to node

        :node: The node id
        :returns: list of node

        """
        return list(nx.neighbors(self._graph, node))

    def occupied(self, position: Tuple[int, int]):
        """
        Check if a node is occupied
        :position: node position
        """
        if position[0] < 0 or position[0] >= self.size_x or position[1] < 0 or position[1] >= self.size_y:
            return True

        return "agent" not in self._graph.nodes[position] or self._graph.nodes[position]["agent"] is not None

    def is_free_from_obs(self, position: Tuple[int, int]):
        if position[0] < 0 or position[0] >= self.size_x or position[1] < 0 or position[1] >= self.size_y:
            return False

        if "obstacle" not in self._graph.nodes[position]:
            return False

        return True

        #if self._graph.nodes[position] == "agent":
            #return True
        #if self._graph.nodes[position] == "obstacle":
            #return False

    def move_agent(self, agent: AgentInterface, new_position: Tuple[int, int]):
        # print(agent.position, new_position)
        if agent.position == new_position:
            print("No move.")
            return

        assert self._graph.nodes[agent.position]["agent"] == agent, \
            f"Error, agent is not currently located at {agent.position}"
        assert self._graph.nodes[new_position]["agent"] is None, \
            f"Error, location is not free {new_position}"

        self._graph.nodes[agent.position]["agent"] = None
        self._graph.nodes[new_position]["agent"] = agent

        agent.position = new_position

    def add_agent_to_map(self, agent: AgentInterface):
        assert self._graph.nodes[agent.position]["agent"] is None, \
            f"Trying to place an agent at a none free location {agent.position}"
        self._graph.nodes[agent.position]["agent"] = agent

    def number_of_nodes(self) -> int:
        """ Get the size of the world """
        return nx.number_of_nodes(self._graph)

    def cost(self, node_1, node_2) -> int:
        """ Get the cost between two nodes

        :node_1: Node id
        :node_2: NOde id
        :returns: The cost

        """
        return self._graph[node_1][node_2]["weight"]

    def get_agents_numbers(self, node):
        """ Get the number of agents and a given node

        :node: The node id
        :returns: The number of agents

        """
        assert 0 <= node <= nx.number_of_nodes(self._graph), "Get number of agents from a non existing node"
        ret = self._graph.nodes[node].get("agents")
        if ret is None:
            return 0

        return ret

    def update_value(self, node: int, key: str, value):
        """ Update a value on a node

        :node: The node id
        :key: The key
        :value: The value to assign

        """
        node = int(node)
        assert 0 <= node <= nx.number_of_nodes(self._graph), "Updating value on a non-existing node"
        self._graph.nodes[node][key] = value

    def view(self, block=True):
        """ Show the world

        """
        color = []

        for _, data in list(self._graph.nodes.data()):
            if "agent" in data:
                if data["agent"] is None:
                    c = FREE
                else:
                    c = AGENT
            else:
                if "obstacle" in data:
                    c = OBSTACLE
                else:
                    assert False, "This should never happen"

            color.append(c)

        #        color = self._graph.nodes(data="agent", default=None)
        #        color = [FREE if c is None else AGENT for _, c in color]

        pos = {n: (n[0] * 10, n[1] * 10) for n in nx.nodes(self._graph)}

        nodes_graph = nx.draw_networkx_nodes(self._graph, pos=pos, node_color=color,
                                             node_size=160, node_shape="s", linewidths=1.0)

        #nodes_graph = nx.draw(self._graph, pos=pos, node_color=color,
        #node_size = 10, node_shape = "s", linewidths = 1.0)

        nodes_graph.set_edgecolor('black')

    def get_map_info(self):
        color = []
        info_map_list = []
        info_map = torch.ones(self.size_x, self.size_y)

        for _, data in list(self._graph.nodes.data()):
            if "agent" in data:
                if data["agent"] is None:
                    c = FREE
                    v = 0
                else:
                    c = AGENT
                    v = 2
            else:
                if "obstacle" in data:
                    c = OBSTACLE
                    v = 1
                else:
                    assert False, "This should never happen"

            color.append(c)
            info_map_list.append(v)

        pos = list(nx.nodes(self._graph))

        for n in range(len(info_map_list)):
            info_map[pos[n]] = info_map_list[n]

        # info_map = torch.tensor(info_map_list)
        # info_map = torch.reshape(info_map,(self.size_x,self.size_y))

        return info_map

    def obstacle(self):
        info_map = torch.zeros(self.size_x, self.size_y)

        for pos, data in list(self._graph.nodes.data()):
            if "obstacle" in data:
                info_map[pos] = 1

        return info_map

    @staticmethod
    def man_dist(a, b):
        (x1, y1) = a
        (x2, y2) = b

        return abs(x1 - x2) + abs(y1 - y2)

    def a_star_map(self, position, goal):
        #return []
        return nx.astar_path(self._graph, position, goal, heuristic=Map.man_dist)
