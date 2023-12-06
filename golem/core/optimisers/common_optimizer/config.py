from copy import deepcopy
from enum import Enum, auto
from functools import partial
from random import random

import numpy as np
from typing import Optional, Tuple, List

from golem.core.optimisers.common_optimizer.common_optimizer import CommonOptimizer, CommonOptimizerParameters
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum
from golem.core.optimisers.initial_graphs_generator import InitialPopulationGenerator
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from test.unit.utils import RandomMetric

graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd', 'e', 'f'])
requirements = GraphRequirements()
objective = Objective({'random_metric': partial(RandomMetric.get_value, delay=0)})
initial_graphs = InitialPopulationGenerator(10, graph_generation_params, requirements)()
graph_optimizer_params = AlgorithmParameters()



class AdaptiveParametersTask(Task):
    def __init__(self, parameters: CommonOptimizerParameters):
        self.pop_size = parameters.graph_generation_params.pop_size

    def update_parameters(self, parameters: CommonOptimizerParameters):
        parameters.graph_generation_params.pop_size = self.pop_size
        return parameters


class Scheme_1(Scheme):
    def __init__(self):
        super().__init__()
        self.__map = {None: {TaskStatusEnum.NEXT: 'adaptive_pop_size'},
                      'adaptive_pop_size': {TaskStatusEnum.SUCCESS: None,
                                            TaskStatusEnum.FAIL: 'adaptive_pop_size'}}

def adaptive_size(task):
    if random() > 0.5:
        task.pop_size += 1
        task.status = TaskStatusEnum.SUCCESS
    else:
        task.pop_size -= 1
        task.status = TaskStatusEnum.FAIL
    return task

pop_size_node = Node('pop_size', adaptive_size)



optimizer = CommonOptimizer(objective=objective,
                            graph_generation_params=graph_generation_params,
                            requirements=requirements,
                            initial_graphs=initial_graphs,
                            graph_optimizer_params=graph_optimizer_params)

pool1 = Pool()


