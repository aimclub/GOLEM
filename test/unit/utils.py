import time
from random import randint
from typing import Sequence, Optional, List, Callable

from golem.core.dag.graph import Graph
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.linked_graph_node import LinkedGraphNode


def nodes_same(left_nodes: Sequence[GraphNode], right_nodes: Sequence[GraphNode]) -> bool:
    left_set = set(map(lambda n: n.descriptive_id, left_nodes))
    right_set = set(map(lambda n: n.descriptive_id, right_nodes))
    return left_set == right_set and len(left_nodes) == len(right_nodes)


def graphs_same(left: Graph, right: Graph) -> bool:
    return left == right


def find_same_node(nodes: List[GraphNode], target: GraphNode) -> Optional[GraphNode]:
    return next(filter(lambda n: n.descriptive_id == target.descriptive_id, nodes), None)


def find_first(graph, predicate: Callable[[GraphNode], bool]) -> Optional[GraphNode]:
    return next(filter(predicate, graph.nodes), None)


def graph_first():
    #    a
    #  |     \
    #  a       b
    # |  \    |  \
    # c   d   c   d
    graph = GraphDelegate()

    root_of_tree, root_child_first, root_child_second = \
        [LinkedGraphNode(oper) for oper in ('a', 'a', 'b')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_oper in ('c', 'd'):
            new_node = LinkedGraphNode(requirement_oper)
            root_node_child.nodes_from.append(new_node)
            graph.add_node(new_node)
        graph.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    graph.add_node(root_of_tree)
    return graph


def graph_second():
    #      a
    #   |      \
    #   a        b
    #  |  \     |  \
    #  c   a    c    d
    #     |  \
    #     b   d

    new_node = LinkedGraphNode('a')
    for oper_type in ('b', 'd'):
        new_node.nodes_from.append(LinkedGraphNode(oper_type))
    graph = graph_first()
    graph.update_subtree(graph.root_node.nodes_from[0].nodes_from[1], new_node)
    return graph


def graph_third():
    #      a
    #   /  |  \
    #  b   d   b
    root_of_tree = LinkedGraphNode('a')
    for oper_type in ('b', 'd', 'b'):
        root_of_tree.nodes_from.append(LinkedGraphNode(oper_type))
    graph = GraphDelegate()

    for node in root_of_tree.nodes_from:
        graph.add_node(node)
    graph.add_node(root_of_tree)

    return graph


def graph_fourth():
    #      a
    #   |  \  \
    #  b   a   b
    #      |  \
    #      b   b

    graph = graph_third()
    new_node = LinkedGraphNode('a')
    [new_node.nodes_from.append(LinkedGraphNode('b')) for _ in range(2)]
    graph.update_subtree(graph.root_node.nodes_from[1], new_node)

    return graph


def graph_fifth():
    # a   b
    #  \ /
    #   c
    #   |
    #   d
    #   |
    #   e
    node_a_primary = LinkedGraphNode('a')
    node_b_primary = LinkedGraphNode('b')

    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary, node_b_primary])
    node_d = LinkedGraphNode('d', nodes_from=[node_c])

    node_e = LinkedGraphNode('e', nodes_from=[node_d])

    graph = GraphDelegate(node_e)
    return graph


def graph_with_single_node():
    node = LinkedGraphNode('a')
    graph = GraphDelegate(node)
    return graph


def simple_linear_graph():
    node_a_primary = LinkedGraphNode('a')
    node_b = LinkedGraphNode('b', nodes_from=[node_a_primary])
    node_c = LinkedGraphNode('c', nodes_from=[node_b])

    graph = GraphDelegate(node_c)
    return graph


def tree_graph():
    # a   b
    #  \ /
    #   c
    #   |
    #   d
    node_a_primary = LinkedGraphNode('a')
    node_b_primary = LinkedGraphNode('b')

    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary, node_b_primary])
    node_d = LinkedGraphNode('d', nodes_from=[node_c])
    graph = GraphDelegate(node_d)
    return graph


class RandomMetric:
    @staticmethod
    def get_value(graph, *args, delay=0, **kwargs) -> float:
        time.sleep(delay)
        return randint(0, 1000)


class DepthMetric:
    @staticmethod
    def get_value(graph: Graph, *args, **kvargs) -> float:
        return graph.depth
