from copy import deepcopy
from random import choice

import numpy as np
import pytest

from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptNode
from test.unit.adapter.graph_data import get_graphs, graph_with_custom_parameters, get_complex_graph, get_adapters, \
    get_optgraphs, networkx_graph_with_parameters
from test.unit.mocks.common_mocks import MockNode, MockAdapter
from test.unit.utils import find_first


@pytest.mark.parametrize('adapter, graph_with_params', [(MockAdapter(), graph_with_custom_parameters),
                                                        (BaseNetworkxAdapter(), networkx_graph_with_parameters)])
def test_adapters_params_correct(adapter, graph_with_params):
    """ Checking the correct conversion of hyperparameters in nodes when nodes
    are passing through adapter
    """
    init_alpha = 12.1
    graph = graph_with_params(init_alpha)

    # Convert into OptGraph object
    opt_graph = adapter.adapt(graph)
    assert np.isclose(init_alpha, opt_graph.root_node.parameters['alpha'])
    # Get graph object back
    restored_graph = adapter.restore(opt_graph)
    # Get hyperparameter value after graph restoration
    if isinstance(graph, Graph):
        restored_alpha = restored_graph.root_node.content['params']['alpha']
    else:
        restored_alpha = restored_graph.nodes['c']['alpha']
    assert np.isclose(init_alpha, restored_alpha)


@pytest.mark.parametrize('adapter', get_adapters())
@pytest.mark.parametrize('optgraph', get_optgraphs())
def test_restored_and_adapted_are_equal(adapter, optgraph):
    graph = adapter.restore(optgraph)
    retransformed_optgraph = adapter.adapt(graph)

    # assert 2-way mapping doesn't change the structure
    assert retransformed_optgraph.descriptive_id == optgraph.descriptive_id
    # assert that new graph is a different object
    assert id(optgraph) != id(retransformed_optgraph)


@pytest.mark.parametrize('adapter', [MockAdapter()])
@pytest.mark.parametrize('graph', get_graphs())
def test_graph_adapt_properly(adapter, graph):
    verifier = GraphVerifier(DEFAULT_DAG_RULES)

    assert all(isinstance(node, MockNode) for node in graph.nodes)
    assert _check_nodes_references_correct(graph)
    assert verifier(graph)

    opt_graph = adapter.adapt(graph)

    assert all(type(node) is OptNode for node in opt_graph.nodes)  # checking strict type equality!
    assert _check_nodes_references_correct(opt_graph)
    assert verifier(opt_graph)


@pytest.mark.parametrize('adapter', [MockAdapter()])
@pytest.mark.parametrize('graph', get_graphs())
def test_adapted_has_same_structure(adapter, graph):
    opt_graph = adapter.adapt(graph)

    # assert graph structures are same
    assert graph.descriptive_id == opt_graph.descriptive_id


@pytest.mark.parametrize('adapter', [MockAdapter()])
@pytest.mark.parametrize('graph', get_graphs())
def test_adapted_and_restored_are_equal(adapter, graph):
    opt_graph = adapter.adapt(graph)
    restored_graph = adapter.restore(opt_graph)

    # assert 2-way mapping doesn't change the structure
    assert graph.descriptive_id == restored_graph.descriptive_id
    # assert that new graph is a different object
    assert id(graph) != id(restored_graph)


@pytest.mark.parametrize('adapter', [MockAdapter()])
@pytest.mark.parametrize('graph', get_graphs())
def test_changes_to_transformed_dont_affect_origin(adapter, graph):
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
    graph = get_complex_graph()
    adapter.adapt(graph)

    assert not find_first(graph, lambda n: type(n) in (GraphNode, OptNode))


def _check_nodes_references_correct(graph):
    for node in graph.nodes:
        if node.nodes_from:
            for parent_node in node.nodes_from:
                if parent_node not in graph.nodes:
                    return False
    return True
