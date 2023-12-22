import random

from .base_experiment import BaseExperiment

from ..swarm import Swarm
from typing import Tuple, Callable
from ..agent import *

from ..map import *


class IdaTestExperiment(BaseExperiment):
    """
    All the basic configuration is inherited from the BaseExperiment. All we need to do is just to override the
    _create_swarm method to ensure that the correct agents are created.
    """

    def _create_swarm(self, args, my_map: Map):
        swarm = Swarm([[args.swarm_size, rl_agent_generator()]], globals()[args.agent_placement], my_map)

        swarm.set_out_path(args.output_path)

        return swarm

