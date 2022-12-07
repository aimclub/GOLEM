from golem.core.dag.graph_utils import nodes_from_layer, distance_to_root_level
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
