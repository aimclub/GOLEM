from copy import deepcopy
from random import choice

import numpy as np
import pytest

from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptNode
from test.unit.adapter.mock_adapter import MockNode, MockDomainStructure, MockAdapter
from test.unit.utils import find_first


def get_graphs():
    node_a = MockNode('operation_a')
    node_b = MockNode('operation_a', nodes_from=[node_a])
    node_c = MockNode('operation_a', nodes_from=[node_b, node_a])
    skip_connection_structure = MockDomainStructure([node_c])

    one_node_graph = MockDomainStructure([node_a])

    node_d = MockNode('operation_a', nodes_from=[node_b])
    linear_graph = MockDomainStructure([node_d])

    node_f = MockNode('operation_f')
    node_g = MockNode('operation_g', nodes_from=[node_b, node_f])
    branching_structure = MockDomainStructure([node_g])

    node_k = MockNode('operation_k', nodes_from=[node_f])
    node_m = MockNode('operation_m', nodes_from=[node_b, node_k])
    node_r = MockNode('operation_r', nodes_from=[node_m])
    branching_structure2 = MockDomainStructure([node_r])

    return [one_node_graph, linear_graph,
            branching_structure, branching_structure2,
            skip_connection_structure]


def graph_with_custom_parameters(alpha_value):
    node_a = MockNode('a')
    node_b = MockNode('b')
    node_c = MockNode('c', nodes_from=[node_a])
    node_d = MockNode('d', nodes_from=[node_b])
    node_final = MockNode('e', nodes_from=[node_c, node_d])
    node_final.content['params'] = {'alpha': alpha_value}
    graph = MockDomainStructure([node_final])

    return graph


def get_complex_graph():
    node_a = MockNode('a')
    node_b = MockNode('b', nodes_from=[node_a])
    node_c = MockNode('c', nodes_from=[node_b])
    node_d = MockNode('d', nodes_from=[node_b, node_c])
    node_e = MockNode('e', nodes_from=[node_d])
    node_final = MockNode('a', nodes_from=[node_c, node_e])
    graph = MockDomainStructure([node_final])
    return graph


def test_adapters_params_correct():
    """ Checking the correct conversion of hyperparameters in nodes when nodes
    are passing through adapter
    """
    init_alpha = 12.1
    graph = graph_with_custom_parameters(init_alpha)

    # Convert into OptGraph object
    adapter = MockAdapter()
    opt_graph = adapter.adapt(graph)
    # Get graph object back
    restored_graph = adapter.restore(opt_graph)
    # Get hyperparameter value after graph restoration
    restored_alpha = restored_graph.root_node.content['params']['alpha']
    assert np.isclose(init_alpha, restored_alpha)


@pytest.mark.parametrize('graph', get_graphs())
def test_graph_adapt_properly(graph):
    adapter = MockAdapter()
    verifier = GraphVerifier(DEFAULT_DAG_RULES)

    assert all(isinstance(node, MockNode) for node in graph.nodes)
    assert _check_nodes_references_correct(graph)
    assert verifier(graph)

    opt_graph = adapter.adapt(graph)

    assert all(type(node) is OptNode for node in opt_graph.nodes)  # checking strict type equality!
    assert _check_nodes_references_correct(opt_graph)
    assert verifier(opt_graph)


@pytest.mark.parametrize('graph', get_graphs())
def test_adapted_has_same_structure(graph):
    adapter = MockAdapter()

    opt_graph = adapter.adapt(graph)

    # assert graph structures are same
    assert graph.descriptive_id == opt_graph.descriptive_id


@pytest.mark.parametrize('graph', get_graphs())
def test_adapted_and_restored_are_equal(graph):
    adapter = MockAdapter()

    opt_graph = adapter.adapt(graph)
    restored_graph = adapter.restore(opt_graph)

    # assert 2-way mapping doesn't change the structure
    assert graph.descriptive_id == restored_graph.descriptive_id
    # assert that new graph is a different object
    assert id(graph) != id(restored_graph)


@pytest.mark.parametrize('graph', get_graphs())
def test_changes_to_transformed_dont_affect_origin(graph):
    adapter = MockAdapter()

    original_graph = deepcopy(graph)
    opt_graph = adapter.adapt(graph)

    # before change they're equal
    assert graph.descriptive_id == opt_graph.descriptive_id

    changed_node = choice(opt_graph.nodes)
    changed_node.content['name'] = 'another_operation'

    # assert that changes to the adapted graph don't affect original graph
    assert graph.descriptive_id != opt_graph.descriptive_id
    assert graph.descriptive_id == original_graph.descriptive_id

    original_opt_graph = deepcopy(opt_graph)
    restored_graph = adapter.restore(opt_graph)

    # before change they're equal
    assert opt_graph.descriptive_id == restored_graph.descriptive_id

    changed_node = choice(restored_graph.nodes)
    changed_node.content['name'] = 'yet_another_operation'

    # assert that changes to the restored graph don't affect original graph
    assert opt_graph.descriptive_id != restored_graph.descriptive_id
    assert opt_graph.descriptive_id == original_opt_graph.descriptive_id


def test_no_opt_or_graph_nodes_after_adapt_so_complex_graph():
    adapter = MockAdapter()
    pipeline = get_complex_graph()
    adapter.adapt(pipeline)

    assert not find_first(pipeline, lambda n: type(n) in (GraphNode, OptNode))


def _check_nodes_references_correct(graph):
    for node in graph.nodes:
        if node.nodes_from:
            for parent_node in node.nodes_from:
                if parent_node not in graph.nodes:
                    return False
    return True
