import pytest

from golem.core.dag.graph_utils import nodes_from_layer, distance_to_root_level, ordered_subnodes_hierarchy
from test.unit.dag.test_graph_operator import get_graph
from test.unit.utils import graph_first
from golem.core.dag.graph_utils import distance_to_primary_level
from golem.core.dag.linked_graph_node import LinkedGraphNode


def get_nodes():
    node_a_first = LinkedGraphNode('a')
    node_a_second = LinkedGraphNode('a')
    node_b = LinkedGraphNode('b', nodes_from=[node_a_first, node_a_second])
    node_d = LinkedGraphNode('d', nodes_from=[node_b])

    return [node_d, node_b, node_a_second, node_a_first]


def test_distance_to_primary_level():
    # given
    root = get_nodes()[0]

    distance = distance_to_primary_level(root)

    assert distance == 2


def test_nodes_from_height():
    graph = graph_first()
    found_nodes = nodes_from_layer(graph, 1)
    true_nodes = [node for node in graph.root_node.nodes_from]
    assert all([node_model == found_node for node_model, found_node in
                zip(true_nodes, found_nodes)])


def test_distance_to_root_level():
    # given
    graph = get_graph()
    selected_node = graph.nodes[2]

    # when
    height = distance_to_root_level(graph, selected_node)

    # then
    assert height == 2


def test_nodes_from_layer():
    # given
    graph = get_graph()
    desired_layer = 2

    # when
    nodes_from_desired_layer = nodes_from_layer(graph, desired_layer)

    # then
    assert len(nodes_from_desired_layer) == 2


def test_ordered_subnodes_hierarchy():
    first_node = LinkedGraphNode('a')
    second_node = LinkedGraphNode('b')
    third_node = LinkedGraphNode('c', nodes_from=[first_node, second_node])
    root = LinkedGraphNode('d', nodes_from=[third_node])

    ordered_nodes = ordered_subnodes_hierarchy(root)

    assert len(ordered_nodes) == 4
    assert ordered_nodes == [root, third_node, first_node, second_node]


def test_ordered_subnodes_cycle():
    cycle_node = LinkedGraphNode('knn')
    second_node = LinkedGraphNode('knn')
    third_node = LinkedGraphNode('lda', nodes_from=[cycle_node, second_node])
    root = LinkedGraphNode('logit', nodes_from=[third_node])
    cycle_node.nodes_from = [root]

    with pytest.raises(ValueError, match='cycle'):
        ordered_subnodes_hierarchy(root)
