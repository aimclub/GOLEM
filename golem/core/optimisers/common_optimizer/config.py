import time
from collections import defaultdict
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from random import random, sample

import numpy as np
from typing import Optional, Tuple, List

from golem.core.optimisers.common_optimizer.common_optimizer import CommonOptimizer, CommonOptimizerParameters
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.runner import OneThreadRunner
from golem.core.optimisers.common_optimizer.scheme import Scheme, SequentialScheme
from golem.core.optimisers.common_optimizer.stage import Stage
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
        super().__init__()
        self.pop_size = parameters.graph_optimizer_params.pop_size

    def update_parameters(self, parameters: CommonOptimizerParameters):
        parameters.graph_optimizer_params.pop_size = self.pop_size
        return parameters

def adaptive_size(task):
    if random() > 0.5:
        task.pop_size += 1
        task.status = TaskStatusEnum.SUCCESS
    else:
        task.pop_size -= 1
        task.status = TaskStatusEnum.FAIL
    return task

def adaptive_parameter_updater(results, parameters):
    return results[0].update_parameters(parameters)

pop_size_node_1 = Node('pop_size1', adaptive_size)
pop_size_node_2 = Node('pop_size2', adaptive_size)
nodes = [pop_size_node_1, pop_size_node_2]

stage1 = Stage(runner=OneThreadRunner(), nodes=nodes, scheme=SequentialScheme(nodes=[x.name for x in nodes]),
               task_builder=AdaptiveParametersTask, stop_fun=lambda f, a: bool(f),
               parameter_updater=adaptive_parameter_updater)

optimizer = CommonOptimizer(objective=objective,
                            graph_generation_params=graph_generation_params,
                            requirements=requirements,
                            initial_graphs=initial_graphs,
                            graph_optimizer_params=graph_optimizer_params)

optimizer.generations = [[f"g{i}" for i in range(10)]]
# res = stage1.run(optimizer.parameters)


class PopulationReproducerTask(Task):
    def __init__(self, parameters: CommonOptimizerParameters):
        super().__init__()
        self.generation = parameters.generations[-1]

def mock_crossover(task):
    inds = sample(task.generation, max(1, round(random() * 10)))
    task.status = TaskStatusEnum.SUCCESS

    tasks = list()
    for ind in inds:
        new_task = task.copy()
        new_task.generation = [ind + 'c']
        tasks.append(new_task)
    return tasks

def mock_mutation(task):
    time.sleep(0.02)
    if random() > 0.5:
        if task.generation[0][-1] == 'c':
            task.generation[0] = task.generation[0] + 'm0'
        parts = task.generation[0].split('m')
        task.generation[0] = 'm'.join(parts[0:-1]) + 'm' + str(int(parts[-1]) + 1)
        task.status = TaskStatusEnum.SUCCESS
    else:
        task.status = TaskStatusEnum.FAIL
    return task

nodes = [Node('crossover', mock_crossover)]
nodes += [Node(f"mutation_{i}", mock_mutation) for i in range(5)]
# TODO some nodes for each variant
scheme_map = {None: defaultdict(lambda: 'crossover'),
              'crossover': {TaskStatusEnum.SUCCESS: 'mutation_0',
                            TaskStatusEnum.FAIL: 'crossover'},
              'mutation_0': {TaskStatusEnum.SUCCESS: 'mutation_1',
                             TaskStatusEnum.FAIL: 'mutation_0'},
              'mutation_1': {TaskStatusEnum.SUCCESS: 'mutation_2',
                             TaskStatusEnum.FAIL: 'mutation_1'},
              'mutation_2': {TaskStatusEnum.SUCCESS: 'mutation_3',
                             TaskStatusEnum.FAIL: 'mutation_2'},
              'mutation_3': {TaskStatusEnum.SUCCESS: 'mutation_4',
                             TaskStatusEnum.FAIL: 'mutation_3'},
              'mutation_4': {TaskStatusEnum.SUCCESS: None,
                             TaskStatusEnum.FAIL: 'mutation_4'}}
def reproducer_parameter_updater(results, parameters):
    parameters.generations.append([res.generation[0] for res in results])
    return parameters

stage2 = Stage(runner=OneThreadRunner(), nodes=nodes, scheme=Scheme(scheme_map),
               task_builder=PopulationReproducerTask, stop_fun=lambda f, a: len(f) > 20,
               parameter_updater=reproducer_parameter_updater)
res = stage2.run(optimizer.parameters)



#
# nodes = [Node('stopper', lambda x: x)]
# def stopper_parameter_updater(results, parameters):
#     if len(parameters.generations) > 3:
#         parameters._run = False
#     return parameters
#
# stage3 = Stage(runner=OneThreadRunner(), nodes=nodes, scheme=SequentialScheme(nodes=[x.name for x in nodes]),
#                task_builder=Task, stop_fun=lambda f, a: bool(f),
#                parameter_updater=stopper_parameter_updater)
# # res = stage3.run(optimizer.parameters)
#
#
#
# optimizer = CommonOptimizer(objective=objective,
#                             graph_generation_params=graph_generation_params,
#                             requirements=requirements,
#                             initial_graphs=initial_graphs,
#                             graph_optimizer_params=graph_optimizer_params,
#                             stages=[stage1, stage2, stage3])
# optimizer.generations = [[f"g{i}" for i in range(10)]]
# optimizer.optimise(1)
#
# print(*optimizer.generations, sep='\n')

