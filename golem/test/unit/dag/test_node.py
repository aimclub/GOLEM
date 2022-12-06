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
