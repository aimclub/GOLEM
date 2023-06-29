from copy import copy
from typing import Sequence, List, TYPE_CHECKING, Callable, Optional

from networkx import simple_cycles

from golem.core.dag.convert import graph_structure_as_nx_graph

if TYPE_CHECKING:
    from golem.core.dag.graph import Graph
    from golem.core.dag.graph_node import GraphNode


def distance_to_root_level(graph: 'Graph', node: 'GraphNode') -> int:
    """Gets distance to the final output node

    Args:
        graph: graph for finding the distance
        node: search starting point

    Return:
        int: distance to root level
    """

    def recursive_child_height(parent_node: 'GraphNode', visited_nodes: Optional[Sequence['GraphNode']] = None) -> int:
        if visited_nodes is None:
            visited_nodes = []
        if parent_node in visited_nodes:
            return -1
        visited_nodes.append(parent_node)
        node_child = graph.node_children(parent_node)
        if node_child:
            height = recursive_child_height(node_child[0], copy(visited_nodes))
            return height + 1 if height >= 0 else -1
        return 0

    height = recursive_child_height(node)
    return height


def distance_to_primary_level(node: 'GraphNode') -> int:
    return node_depth(node) - 1 if node_depth(node) > 0 else -1


def nodes_from_layer(graph: 'Graph', layer_number: int) -> Sequence['GraphNode']:
    """Gets all the nodes from the chosen layer up to the surface

    Args:
        graph: graph with nodes
        layer_number: max height of diving

    Returns:
        all nodes from the surface to the ``layer_number`` layer
    """

    def get_nodes(roots: Sequence['GraphNode'], current_height: int) -> Sequence['GraphNode']:
        """Gets all the parent nodes of ``roots``

        :param roots: nodes to get all subnodes from
        :param current_height: current diving step depth

        :return: all parent nodes of ``roots`` in one sequence:69
        """
        nodes = []
        if current_height == layer_number:
            nodes.extend(roots)
        else:
            for root in roots:
                nodes.extend(get_nodes(root.nodes_from, current_height + 1))
        return nodes

    nodes = get_nodes(graph.root_nodes(), current_height=0)
    return nodes


def ordered_subnodes_hierarchy(node: 'GraphNode') -> List['GraphNode']:
    """Gets hierarchical subnodes representation of the graph starting from the bounded node

    Returns:
        List['GraphNode']: hierarchical subnodes list starting from the bounded node
    """
    started = {node}
    visited = set()

    def subtree_impl(node):
        nodes = [node]
        for parent in node.nodes_from:
            if parent in visited:
                continue
            elif parent in started:
                raise ValueError('Can not build ordered node hierarchy: graph has cycle')
            started.add(parent)
            nodes.extend(subtree_impl(parent))
            visited.add(parent)
        return nodes

    return subtree_impl(node)


def node_depth(node: 'GraphNode', visited_nodes: Optional[Sequence['GraphNode']] = None) -> int:
    """Gets this graph depth from the provided ``node`` to the graph source node

    Args:
        node: where to start diving from


    Returns:
        int: length of a path from the provided ``node`` to the farthest primary node
    """
    if visited_nodes is None:
        visited_nodes = []
    if node in visited_nodes:
        return -1

    elif not node.nodes_from:
        return 1
    else:
        visited_nodes.append(node)
        parent_depths = [node_depth(next_node, copy(visited_nodes)) for next_node in node.nodes_from]
        if any([depth < 0 for depth in parent_depths]):
            return -1
        else:
            return 1 + max(parent_depths)


def map_dag_nodes(transform: Callable, nodes: Sequence) -> Sequence:
    """Maps nodes in dfs-order while respecting node edges.

    Args:
        transform: node transform function (maps node to node)
        nodes: sequence of nodes for mapping

    Returns:
        Sequence: sequence of transformed links with preserved relations
    """
    mapped_nodes = {}

    def map_impl(node):
        already_mapped = mapped_nodes.get(id(node))
        if already_mapped:
            return already_mapped
        # map node itself
        mapped_node = transform(node)
        # remember it to avoid recursion
        mapped_nodes[id(node)] = mapped_node
        # map its children
        mapped_node.nodes_from = list(map(map_impl, node.nodes_from))
        return mapped_node

    return list(map(map_impl, nodes))


def graph_structure(graph: 'Graph') -> str:
    """ Returns structural information about the graph - names and parameters of graph nodes.
    Represents graph info in easily readable way.

    Returns:
        str: graph structure
    """
    return '\n'.join([str(graph), *(f'{node.name} - {node.parameters}' for node in graph.nodes)])


def graph_has_cycle(graph: 'Graph') -> bool:
    """ Returns True if the graph contains a cycle and False otherwise."""
    nx_graph, _ = graph_structure_as_nx_graph(graph)
    return len(list(simple_cycles(nx_graph))) > 0
