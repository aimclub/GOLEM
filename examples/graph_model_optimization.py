import datetime
import os
import random
from typing import Type

import numpy as np
import pandas as pd

from golem.core.adapter import DirectAdapter
from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.regularization import RegularizationTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.core.optimisers.random.random_mutation_optimizer import RandomMutationOptimizer
from golem.core.optimisers.random.random_search import RandomSearchOptimizer
from golem.core.paths import project_root

random.seed(1)
np.random.seed(1)


class CustomGraphModel(OptGraph):
    def evaluate(self, data: pd.DataFrame):
        nodes = data.columns.to_list()
        _, labels = graph_structure_as_nx_graph(self)
        return len(nodes)


class CustomGraphNode(OptNode):
    def __str__(self):
        return f'Node_{self.name}'


def custom_metric(graph: CustomGraphModel, data: pd.DataFrame, visualisation: bool = False):
    if visualisation:
        graph.show()
    existing_variables_num = -graph.depth - graph.evaluate(data)

    return existing_variables_num


def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True


def custom_mutation(graph: CustomGraphModel, **kwargs) -> CustomGraphModel:
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(graph.length))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            graph.connect_nodes(random_node, other_random_node)
    except Exception as ex:
        default_log(prefix='custom_mutation').warning(f'Incorrect connection: {ex}')
    return graph


def run_custom_example(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer, timeout: datetime.timedelta = None,
                       visualisation: bool = True):
    if not timeout:
        timeout = datetime.timedelta(minutes=3)

    data = pd.read_csv(os.path.join(project_root(), 'examples', 'data', 'custom_encoded.csv'))
    nodes_types = ['V1', 'V2', 'V3',
                   'V4', 'V5', 'V6',
                   'V7', 'V8', 'V9', 'V10']

    initial = [CustomGraphModel(nodes=[CustomGraphNode(node_type) for node_type in nodes_types])]

    requirements = GraphRequirements(
        max_arity=10,
        max_depth=10,
        timeout=timeout)

    optimizer_parameters = GPAlgorithmParameters(
        pop_size=5,
        crossover_prob=0.8, mutation_prob=0.9,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[custom_mutation],
        crossover_types=[CrossoverTypesEnum.none],
        regularization_type=RegularizationTypesEnum.none)

    adapter = DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode)
    constraints = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates]
    graph_generation_params = GraphGenerationParams(
        adapter=adapter,
        rules_for_constraint=constraints,
        available_node_types=nodes_types)

    objective = Objective({'custom': custom_metric})

    optimizer = optimizer_cls(
        objective=objective,
        initial_graphs=initial,
        requirements=requirements,
        graph_generation_params=graph_generation_params,
        graph_optimizer_params=optimizer_parameters)

    objective_eval = ObjectiveEvaluate(objective, data=data, visualisation=visualisation)
    optimized_graphs = optimizer.optimise(objective_eval)
    optimized_network = adapter.restore(optimized_graphs[0])
    if visualisation:
        optimized_network.show()


if __name__ == '__main__':
    visualisation = False
    timeout = datetime.timedelta(minutes=1)
    optimizers = [EvoGraphOptimizer, RandomSearchOptimizer, RandomMutationOptimizer]
    for optimizer_cls in optimizers:
        run_custom_example(optimizer_cls, timeout, visualisation)
