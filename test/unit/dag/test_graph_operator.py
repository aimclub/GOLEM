from copy import deepcopy

import pytest

from golem.core.adapter import DirectAdapter
from golem.core.dag.graph_delegate import GraphDelegate
from golem.core.dag.linked_graph import LinkedGraph, get_distance_between
from golem.core.dag.linked_graph_node import LinkedGraphNode


@pytest.fixture()
def graph() -> GraphDelegate:
    third_level_one = LinkedGraphNode('l3_n1')

    second_level_one = LinkedGraphNode('l2_n1', nodes_from=[third_level_one])
    second_level_two = LinkedGraphNode('l2_n2')

    first_level_one = LinkedGraphNode('l1_n1', nodes_from=[second_level_one, second_level_two])

    root = LinkedGraphNode('l0_n1', nodes_from=[first_level_one])
    graph = GraphDelegate(root)

    return graph


def test_graph_operator_init(graph):
    assert type(graph.operator) is LinkedGraph


def test_actualise_old_node_children(graph):
    # given
    selected_node = graph.nodes[2]
    new_node = LinkedGraphNode('new_node')

    # when
    graph.operator.actualise_old_node_children(old_node=selected_node,
                                               new_node=new_node)
    updated_parent = graph.nodes[1]

    # then
    assert new_node in updated_parent.nodes_from


def test_sort_nodes(graph):
    # given
    selected_node = graph.nodes[2]
    original_length = graph.length
    new_node = LinkedGraphNode('new_n1')
    new_subroot = LinkedGraphNode('new_n2', nodes_from=[new_node])

    # when
    selected_node.nodes_from.append(new_subroot)
    graph.operator.sort_nodes()

    # then
    assert graph.length == original_length + 2
    assert graph.nodes[4] is new_subroot
    assert graph.nodes[5] is new_node


def test_node_children(graph):
    # given
    selected_node = graph.nodes[2]

    # when
    children = graph.node_children(selected_node)

    # then
    assert len(children) == 1
    assert children[0] is graph.nodes[1]


# ------------------------------------------------------------------------------
# Tests for distance_to_other method

def test_distance_to_same_graph_restored(graph):
    # given
    adapter = DirectAdapter()
    opt_graph = adapter.adapt(graph)

    # when
    distance = get_distance_between(graph_1=graph, graph_2=adapter.restore(opt_graph))

    # then
    assert distance == 0


def test_known_distances():
    node_a = LinkedGraphNode('a')
    node_b = LinkedGraphNode('b')
    node_c = LinkedGraphNode('c', nodes_from=[node_a])
    node_c_with_alt_params = LinkedGraphNode(content={'name': 'c', 'params': {'alpha': 4}}, nodes_from=[node_a])
    node_d = LinkedGraphNode('d', nodes_from=[node_a])
    graph_a = GraphDelegate(node_a)  # a
    graph_b = GraphDelegate(node_b)  # b
    graph_c = GraphDelegate(node_c)  # a -> c
    graph_d = GraphDelegate(node_d)  # a -> d
    graph_c_with_alt_params = GraphDelegate(node_c_with_alt_params)  # a -> c_alt_params

    assert get_distance_between(graph_1=graph_c,
                                graph_2=graph_c) == 0  # the same graph
    assert get_distance_between(graph_1=graph_c,
                                graph_2=graph_a) == 2  # changes: 1 node (operation) + 1 edge
    assert get_distance_between(graph_1=graph_c,
                                graph_2=graph_d) == 1  # changes: 1 node (operation)
    assert get_distance_between(graph_1=graph_c,
                                graph_2=graph_c_with_alt_params) == 1  # changes: 1 node (params)
    assert get_distance_between(graph_1=graph_c,
                                graph_2=graph_b) == 3  # changes: 2 nodes (operations) + 1 edge
    assert get_distance_between(graph_1=graph_c,
                                graph_2=graph_c_with_alt_params) == 1  # changes: 1 operation + params


# ------------------------------------------------------------------------------
# Tests for disconnect_nodes method

def get_initial_graph():
    node_a_primary = LinkedGraphNode('a')

    node_b = LinkedGraphNode('b', nodes_from=[node_a_primary])
    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary])
    node_c_second = LinkedGraphNode('c', nodes_from=[node_a_primary])

    node_d = LinkedGraphNode('d', nodes_from=[node_c_second])
    node_e = LinkedGraphNode('e', nodes_from=[node_b, node_c])

    node_e_root = LinkedGraphNode('e', nodes_from=[node_d, node_e])

    graph = GraphDelegate(node_e_root)

    return graph


def get_res_graph_test_first():
    node_a_primary = LinkedGraphNode('a')

    node_c_second = LinkedGraphNode('c', nodes_from=[node_a_primary])

    node_d = LinkedGraphNode('d', nodes_from=[node_c_second])

    node_e_root = LinkedGraphNode('e', nodes_from=[node_d])

    graph = GraphDelegate(node_e_root)

    return graph


def get_res_graph_test_second():
    node_a_primary = LinkedGraphNode('a')

    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary])
    node_c_second = LinkedGraphNode('c', nodes_from=[node_a_primary])

    node_d = LinkedGraphNode('d', nodes_from=[node_c_second])
    node_e = LinkedGraphNode('e', nodes_from=[node_c])

    node_e_root = LinkedGraphNode('e', nodes_from=[node_d, node_e])

    graph = GraphDelegate(node_e_root)

    return graph


def get_res_graph_test_third():
    node_a_primary = LinkedGraphNode('a')

    node_b = LinkedGraphNode('b', nodes_from=[node_a_primary])
    node_c = LinkedGraphNode('c', nodes_from=[node_a_primary])

    node_e = LinkedGraphNode('e', nodes_from=[node_b, node_c])

    node_e_root = LinkedGraphNode('e', nodes_from=[node_e])

    graph = GraphDelegate(node_e_root)

    return graph


def test_disconnect_nodes_method_first():
    graph = get_initial_graph()

    res_graph = get_res_graph_test_first()

    node_e = graph.nodes[4]
    node_e_root = graph.nodes[0]

    graph.disconnect_nodes(node_e, node_e_root, clean_up_leftovers=True)

    assert res_graph == graph


def test_disconnect_nodes_method_second():
    graph = get_initial_graph()

    res_graph = get_res_graph_test_second()

    node_b = graph.nodes[5]
    node_e = graph.nodes[4]

    graph.disconnect_nodes(node_b, node_e, clean_up_leftovers=True)

    assert res_graph == graph


def test_disconnect_nodes_method_third():
    graph = get_initial_graph()

    res_graph = get_res_graph_test_third()

    node_d = graph.nodes[1]
    root_node_e = graph.nodes[0]

    graph.disconnect_nodes(node_d, root_node_e, clean_up_leftovers=True)

    assert res_graph == graph


def test_disconnect_nodes_method_fourth():
    graph = get_initial_graph()

    # Try to disconnect nodes between which there is no edge
    res_graph = deepcopy(graph)

    node_c = res_graph.nodes[2]
    root_node_e = res_graph.nodes[0]

    res_graph.disconnect_nodes(node_c, root_node_e, clean_up_leftovers=True)
    assert res_graph == graph


def test_disconnect_nodes_method_fifth():
    graph = get_initial_graph()

    # Try to disconnect nodes that are not in this graph
    res_graph = deepcopy(graph)

    node_k = LinkedGraphNode('k')
    node_m = LinkedGraphNode('m', nodes_from=[node_k])

    res_graph.disconnect_nodes(node_k, node_m, clean_up_leftovers=True)
    assert res_graph == graph


# ------------------------------------------------------------------------------
# Test for get_edges method

def test_get_edges(graph):
    print(graph.nodes)

    l3_n1 = graph.nodes[3]
    l2_n1 = graph.nodes[2]
    l2_n2 = graph.nodes[4]
    l1_n1 = graph.nodes[1]
    l0_n1 = graph.nodes[0]

    res_edges = [(l1_n1, l0_n1), (l2_n1, l1_n1), (l2_n2, l1_n1), (l3_n1, l2_n1)]

    edges = graph.get_edges()
    assert res_edges == edges


def test_reset_descriptive_id():
    """ Checks if descriptive_id is set to None after any changes in graph. """
    graph = get_initial_graph()

    initial_descriptive_id = graph.descriptive_id
    assert initial_descriptive_id is not None

    graph.delete_node(graph.nodes[0])

    final_descriptive_id = graph.descriptive_id
    assert initial_descriptive_id is not None
    assert final_descriptive_id != initial_descriptive_id
