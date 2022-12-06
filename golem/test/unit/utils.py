from random import randint
from typing import Sequence, Optional, List

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


def graph_first():
    node_a_primary = LinkedGraphNode('a')
    node_b_primary = LinkedGraphNode('b')
    node_c_primary = LinkedGraphNode('c')
    node_d_primary = LinkedGraphNode('d')

    node_e = LinkedGraphNode('e', nodes_from=[node_a_primary, node_b_primary])
    node_f = LinkedGraphNode('f', nodes_from=[node_c_primary, node_d_primary])

    node_g_root = LinkedGraphNode('g', nodes_from=[node_e, node_f])

    graph = GraphDelegate(node_g_root)
    return graph


def graph_second():
    node_a_primary = LinkedGraphNode('a')
    node_b_primary = LinkedGraphNode('b')

    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary, node_b_primary])
    node_d_primary = LinkedGraphNode('d')

    node_e = LinkedGraphNode('e', nodes_from=[node_c, node_d_primary])

    node_f = LinkedGraphNode('f', nodes_from=[node_e])

    node_e_root = LinkedGraphNode('e', nodes_from=[node_f])

    graph = GraphDelegate(node_e_root)
    return graph


def graph_third():
    node_a_primary = LinkedGraphNode('a')
    node_b = LinkedGraphNode('b', nodes_from=[node_a_primary])
    node_c = LinkedGraphNode('c', nodes_from=[node_b])
    node_d = LinkedGraphNode('d', nodes_from=[node_c])
    graph = GraphDelegate(node_d)
    return graph


def graph_fourth():
    node_a_primary = LinkedGraphNode('a')
    node_b_primary = LinkedGraphNode('b')

    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary])
    node_d = LinkedGraphNode('d', nodes_from=[node_b_primary])

    node_e = LinkedGraphNode('e', nodes_from=[node_c, node_d])

    node_f = LinkedGraphNode('f', nodes_from=[node_e])

    graph = GraphDelegate(node_f)
    return graph


def graph_fifth():
    node_a_primary = LinkedGraphNode('a')
    node_b_primary = LinkedGraphNode('b')

    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary, node_b_primary])
    node_d = LinkedGraphNode('d', nodes_from=[node_c])

    node_e = LinkedGraphNode('e', nodes_from=[node_d])

    graph = GraphDelegate(node_e)
    return graph


class RandomMetric:
    @staticmethod
    def get_value(*args, **kvargs) -> float:
        return randint(0, 1000)


class DepthMetric:
    @staticmethod
    def get_value(graph: Graph, *args, **kvargs) -> float:
        return graph.depth
