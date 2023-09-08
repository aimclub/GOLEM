from random import choice

from golem.core.optimisers.genetic.operators.base_mutations import single_drop_mutation, add_as_child, \
    add_separate_parent_node, add_intermediate_node
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters


# mutations which rewards are changing during the optimization process


def fake_add_mutation(graph: OptGraph,
                      requirements: GraphRequirements,
                      graph_gen_params: GraphGenerationParams,
                      parameters: AlgorithmParameters) -> OptGraph:
    """
    Fickle mutation for an experiment. The evolution process in experiment is split in 3 steps:
    On the first step with the help of MAB the `fake_add_mutation` must be chosen,
    since it gives the highest reward, then `fake_add_mutation2` and on the last step -- `fake_add_mutation3`.

    On the first step of evolution this mutation gives reward=3, then reward=-1 and finally reward=1
    """

    if graph.depth >= requirements.max_depth:
        # add mutation is not possible
        return graph

    if len(graph.nodes) < 80:
        nodes_to_add_num = 3
    elif 80 < len(graph.nodes) < 140:
        nodes_to_add_num = -1
    else:
        nodes_to_add_num = 1

    return _apply_mutation(graph=graph, nodes_to_add_num=nodes_to_add_num, requirements=requirements,
                           graph_gen_params=graph_gen_params, parameters=parameters)


def fake_add_mutation2(graph: OptGraph,
                       requirements: GraphRequirements,
                       graph_gen_params: GraphGenerationParams,
                       parameters: AlgorithmParameters) -> OptGraph:
    """
    Fickle mutation for an experiment. The evolution process in experiment is split in 3 steps:
    On the first step with the help of MAB the `fake_add_mutation` must be chosen,
    since it gives the highest reward, then `fake_add_mutation2` and on the last step -- `fake_add_mutation3`.

    On the first step of evolution this mutation gives reward=-1, then reward=2 and finally reward=-1
    """

    if graph.depth >= requirements.max_depth:
        # add mutation is not possible
        return graph

    if len(graph.nodes) < 80:
        nodes_to_add_num = -1
    elif 80 < len(graph.nodes) < 140:
        nodes_to_add_num = 2
    else:
        nodes_to_add_num = -1

    return _apply_mutation(graph=graph, nodes_to_add_num=nodes_to_add_num, requirements=requirements,
                           graph_gen_params=graph_gen_params, parameters=parameters)


def fake_add_mutation3(graph: OptGraph,
                       requirements: GraphRequirements,
                       graph_gen_params: GraphGenerationParams,
                       parameters: AlgorithmParameters) -> OptGraph:
    """
    Fickle mutation for an experiment. The evolution process in experiment is split in 3 steps:
    On the first step with the help of MAB the `fake_add_mutation` must be chosen,
    since it gives the highest reward, then `fake_add_mutation2` and on the last step -- `fake_add_mutation3`.

    On the first step of evolution this mutation gives reward=1, then reward=-3 and finally reward=2
    """

    if graph.depth >= requirements.max_depth:
        # add mutation is not possible
        return graph

    if len(graph.nodes) < 80:
        nodes_to_add_num = 1
    elif 80 < len(graph.nodes) < 140:
        nodes_to_add_num = -3
    else:
        nodes_to_add_num = 2

    return _apply_mutation(graph=graph, nodes_to_add_num=nodes_to_add_num, requirements=requirements,
                           graph_gen_params=graph_gen_params, parameters=parameters)


def _apply_mutation(graph: OptGraph, nodes_to_add_num: int, requirements: GraphRequirements,
                    graph_gen_params: GraphGenerationParams, parameters: AlgorithmParameters):
    """ Adds or deletes nodes according to specified number. """
    if nodes_to_add_num < 0:
        for _ in range(nodes_to_add_num):
            graph = single_drop_mutation(graph, requirements, graph_gen_params, parameters)
    else:
        for _ in range(nodes_to_add_num):
            node_to_mutate = choice(graph.nodes)

            single_add_strategies = [add_as_child, add_separate_parent_node]
            if node_to_mutate.nodes_from:
                single_add_strategies.append(add_intermediate_node)
            strategy = choice(single_add_strategies)

            graph = strategy(graph, node_to_mutate, graph_gen_params.node_factory)

    return graph
