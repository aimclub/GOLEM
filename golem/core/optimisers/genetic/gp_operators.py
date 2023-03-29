from copy import deepcopy
from typing import Any, List, Tuple

from golem.core.dag.graph_utils import distance_to_primary_level


def equivalent_subtree(graph_first: Any, graph_second: Any) -> List[Tuple[Any, Any]]:
    """Finds the similar subtree in two given trees"""

    def structural_equivalent_nodes(node_first, node_second):
        nodes = []
        is_same_type = type(node_first) == type(node_second)
        # check if both nodes are primary or secondary
        # TODO: use normal overriding of __eq__ method instead of instance check
        if hasattr(node_first, 'is_primary') and hasattr(node_second, 'is_primary'):
            is_same_graph_node_type = node_first.is_primary == node_second.is_primary
            is_same_type = is_same_type and is_same_graph_node_type

        if is_same_type and len(node_first.nodes_from) == len(node_second.nodes_from):
            nodes.append((node_first, node_second))
            for node1_child, node2_child in zip(node_first.nodes_from, node_second.nodes_from):
                nodes_set = structural_equivalent_nodes(node1_child, node2_child)
                nodes.extend(nodes_set)
        return nodes

    pairs_set = []
    for root_first, root_second in zip(graph_first.root_nodes(), graph_second.root_nodes()):
        equivalent_pairs = structural_equivalent_nodes(root_first, root_second)
        pairs_set.extend(equivalent_pairs)
    return pairs_set


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
