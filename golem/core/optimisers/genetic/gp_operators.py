import itertools
from copy import deepcopy
from typing import Any, List, Tuple

from golem.core.dag.graph_node import descriptive_id_recursive_nodes
from golem.core.dag.graph_utils import distance_to_primary_level


def equivalent_subtree(graph_first: Any, graph_second: Any, with_primary_nodes: bool = False) \
        -> List[Tuple[Any, Any]]:
    """Finds the similar subtrees in two given trees.
    With `with_primary_nodes` primary nodes are considered too.
    Due to a lot of common subgraphs consisted only of single primary nodes, these nodes can be
    not considered with `with_primary_nodes=False`."""

    def drop_duplicates(match_set: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        unique_match_set = []
        for match in match_set:
            if (match or reversed(match)) not in unique_match_set:
                unique_match_set.append(match)
        return unique_match_set

    def structural_equivalent_nodes(node_first: Any, node_second: Any) -> list:
        def are_subtrees_the_same(match_set: List[Tuple[Any, Any]], node_first: Any, node_second: Any) -> bool:
            """ Returns `True` if subtrees of specified root nodes are the same, otherwise returns `False`. """
            matched = []

            # 1. Number of exact children must be the same
            # 2. All children from one node must have a match from other node children
            # 3. Protection from cycles when lengths of descriptive ids are the same due to cycles
            if len(node_first.nodes_from) != len(node_second.nodes_from) or \
                    len(match_set) == 0 and len(node_first.nodes_from) != 0 or \
                    len(list(set(descriptive_id_recursive_nodes(node_first)))) != \
                    len(list(set(descriptive_id_recursive_nodes(node_second)))):
                return False

            for node in node_first.nodes_from:
                for node2 in node_second.nodes_from:
                    if (node, node2) or (node2, node) in match_set:
                        matched.append((node, node2))
            if len(matched) >= len(node_first.nodes_from):
                return True
            return False

        nodes = []
        is_same_type = type(node_first) == type(node_second)
        # check if both nodes are primary or secondary
        if hasattr(node_first, 'is_primary') and hasattr(node_second, 'is_primary'):
            is_same_graph_node_type = node_first.is_primary == node_second.is_primary
            is_same_type = is_same_type and is_same_graph_node_type

        for node1_child, node2_child in itertools.product(node_first.nodes_from, node_second.nodes_from):
            nodes_set = structural_equivalent_nodes(node1_child, node2_child)
            nodes.extend(nodes_set)
        if is_same_type and len(node_first.nodes_from) == len(node_second.nodes_from) \
                and are_subtrees_the_same(match_set=nodes, node_first=node_first, node_second=node_second):
            nodes.append((node_first, node_second))
        return nodes

    pairs_set = []
    for node_first in graph_first.nodes:
        for node_second in graph_second.nodes:
            if (node_first, node_second) in pairs_set:
                continue
            equivalent_pairs = structural_equivalent_nodes(node_first, node_second)
            pairs_set.extend(equivalent_pairs)

    pairs_set = drop_duplicates(pairs_set)
    if with_primary_nodes:
        return pairs_set
    # remove nodes with no children
    result = []
    for pair in pairs_set:
        if len(pair[0].nodes_from) != 0:
            result.append(pair)
    return result


def replace_subtrees(graph_first: Any, graph_second: Any, node_from_first: Any, node_from_second: Any,
                     layer_in_first: int, layer_in_second: int, max_depth: int):
    node_from_graph_first_copy = deepcopy(node_from_first)

    summary_depth = layer_in_first + distance_to_primary_level(node_from_second) + 1
    if summary_depth <= max_depth and summary_depth != 0:
        graph_first.update_subtree(node_from_first, node_from_second)

    summary_depth = layer_in_second + distance_to_primary_level(node_from_first) + 1
    if summary_depth <= max_depth and summary_depth != 0:
        graph_second.update_subtree(node_from_second, node_from_graph_first_copy)


def num_of_parents_in_crossover(num_of_final_inds: int) -> int:
    return num_of_final_inds if not num_of_final_inds % 2 else num_of_final_inds + 1


def filter_duplicates(archive, population) -> List[Any]:
    filtered_archive = []
    for ind in archive.items:
        has_duplicate_in_pop = False
        for pop_ind in population:
            if ind.fitness == pop_ind.fitness:
                has_duplicate_in_pop = True
                break
        if not has_duplicate_in_pop:
            filtered_archive.append(ind)
    return filtered_archive
