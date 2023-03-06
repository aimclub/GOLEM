import os
from copy import deepcopy
from typing import List, Tuple, Any

from golem.core.dag.linked_graph import get_distance_between
from golem.core.optimisers.graph import OptGraph


def graph_gluing(last_population: List[OptGraph]) -> Tuple[OptGraph, List[List[int]]]:
    """ Glue together the three most distant pipelines from the last population """
    graphs_to_glue = get_graphs_to_glue(population=last_population)
    inds_idxs = [last_population.index(graph) for graph in graphs_to_glue]
    if graphs_to_glue[0].root_node.nodes_from is None:
        graphs_to_glue[0].root_node.nodes_from = []
    graph = deepcopy(graphs_to_glue[0])
    for i in range(1, len(graphs_to_glue)):
        root_of_tree = graph.root_node
        root_of_i_tree = graphs_to_glue[i].root_node
        root_of_tree.nodes_from.append(root_of_i_tree)
        graph.nodes.extend(graphs_to_glue[i].nodes)

    return graph, [inds_idxs]


def get_graphs_to_glue(population: List[OptGraph]) -> List[OptGraph]:
    """ Get three pipelines: the first one is the pipeline from the individual with the best fitness,
    the second one is the most different pipeline from it and
    the third one is pipeline in the middle in the distance from the two previously named pipelines """
    pipe_0 = population[0]
    distances = []
    for idx, ind in enumerate(population):
        cur_distance = get_distance_between(pipe_0, ind)
        distances.append((idx, cur_distance))
    distances.sort(key=lambda x: x[1], reverse=True)
    graphs_to_glue = []
    try:
        graphs_to_glue.append(population[distances[0][0]])
        graphs_to_glue.append(population[distances[len(distances) // 2][0]])
    except IndexError:
        return graphs_to_glue
    return graphs_to_glue


def _get_available_operations(task_type: Any, mode: str, forbidden_operations: List['str']):
    """ Get available operations for the task and exclude forbidden ones """
    return
