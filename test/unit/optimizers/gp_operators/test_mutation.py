from copy import deepcopy
from typing import Sequence, Optional

import numpy as np
import pytest

from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters, MutationStrengthEnum
from golem.core.optimisers.genetic.operators.base_mutations import (
    no_mutation,
    simple_mutation,
    reduce_mutation,
    single_edge_mutation,
    single_drop_mutation,
    single_change_mutation,
    add_separate_parent_node,
    add_as_child,
    add_intermediate_node,
    MutationTypesEnum
)
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from test.unit.utils import simple_linear_graph, tree_graph, graph_with_single_node, graph_first, \
    graph_fifth, simple_cycled_graph

available_node_types = ['a', 'b', 'c', 'd', 'e', 'f']


def get_mutation_params(mutation_types: Sequence[MutationTypesEnum] = None,
                        requirements: Optional[GraphRequirements] = None,
                        mutation_prob: float = 1.0,
                        mutation_strength: Optional[MutationStrengthEnum] = MutationStrengthEnum.mean):
    requirements = requirements or GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=available_node_types)
    mutation_types = mutation_types or (MutationTypesEnum.simple,
                                        MutationTypesEnum.reduce,
                                        MutationTypesEnum.growth,
                                        MutationTypesEnum.local_growth)
    parameters = GPAlgorithmParameters(mutation_types=mutation_types,
                                       mutation_prob=mutation_prob,
                                       mutation_strength=mutation_strength)
    return dict(requirements=requirements,
                graph_gen_params=graph_generation_params,
                parameters=parameters)


def test_mutation_none():
    graph = simple_linear_graph()
    new_graph = deepcopy(graph)
    new_graph = no_mutation(new_graph)
    assert new_graph == graph


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_simple_mutation(graph):
    """
    Test correctness of simple mutation
    """
    new_graph = deepcopy(graph)
    new_graph = simple_mutation(new_graph, **get_mutation_params())
    for i in range(len(graph.nodes)):
        assert graph.nodes[i] != new_graph.nodes[i]


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_drop_node(graph):
    new_graph = deepcopy(graph)
    params = get_mutation_params()
    for _ in range(5):
        new_graph = single_drop_mutation(new_graph, **params)
    assert new_graph.length < graph.length


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_add_as_parent_node(graph):
    """
    Test correctness of adding as a parent
    """
    new_graph = deepcopy(graph)
    params = get_mutation_params()
    node_factory = params['graph_gen_params'].node_factory

    prev_nodes = new_graph.nodes[:]
    add_separate_parent_node(new_graph, node_factory)
    new_nodes = [node for node in new_graph.nodes if node not in prev_nodes]

    assert len(new_nodes) == 1
    assert not new_nodes[0].nodes_from
    assert new_graph.node_children(new_nodes[0])
    assert new_graph.length > graph.length


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_add_as_child_node(graph):
    """
    Test correctness of adding as a child
    """
    new_graph = deepcopy(graph)
    params = get_mutation_params()
    node_factory = params['graph_gen_params'].node_factory

    prev_nodes = new_graph.nodes[:]
    add_as_child(new_graph, node_factory)
    new_nodes = [node for node in new_graph.nodes if node not in prev_nodes]

    assert len(new_nodes) == 1
    assert new_nodes[0].nodes_from
    assert new_graph.length > graph.length


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_add_as_intermediate_node(graph):
    """
    Test correctness of adding as an intermediate node
    """
    new_graph = deepcopy(graph)
    params = get_mutation_params()
    node_factory = params['graph_gen_params'].node_factory
    prev_nodes = new_graph.nodes[:]
    add_intermediate_node(new_graph, node_factory)
    new_nodes = [node for node in new_graph.nodes if node not in prev_nodes]

    assert len(new_nodes) == 1
    assert new_nodes[0].nodes_from
    assert new_graph.node_children(new_nodes[0])
    assert new_graph.length > graph.length


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_edge_mutation_for_graph(graph):
    """
    Tests edge mutation can add edge between nodes
    """
    new_graph = deepcopy(graph)
    new_graph = single_edge_mutation(new_graph, **get_mutation_params())
    assert len(new_graph.get_edges()) > len(graph.get_edges())


@pytest.mark.parametrize('graph', [simple_linear_graph(), tree_graph(), simple_cycled_graph()])
def test_replace_mutation(graph):
    """
    Tests single_change mutation can change node to another
    """
    new_graph = single_change_mutation(graph, **get_mutation_params())
    operations = [node.content['name'] for node in new_graph.nodes]

    assert np.all([operation in available_node_types for operation in operations])


def test_mutation_with_single_node():
    adapter = DirectAdapter()
    graph = adapter.adapt(graph_with_single_node())
    new_graph = deepcopy(graph)
    params = get_mutation_params()

    new_graph = reduce_mutation(new_graph, **params)

    assert graph == new_graph

    new_graph = single_drop_mutation(new_graph, **params)

    assert graph == new_graph


@pytest.mark.parametrize('mutation_type', MutationTypesEnum)
def test_mutation_with_zero_prob(mutation_type):
    adapter = DirectAdapter()
    params = get_mutation_params([mutation_type], mutation_prob=0)
    mutation = Mutation(**params)

    ind = Individual(adapter.adapt(graph_first()))
    new_ind = mutation(ind)

    assert new_ind.graph == ind.graph
    assert new_ind.uid == ind.uid

    ind = Individual(adapter.adapt(graph_fifth()))
    new_ind = mutation(ind)

    assert new_ind.graph == ind.graph
    assert new_ind.uid == ind.uid


def test_mutation_with_max_prob():
    """ Checks that individual is not included in next population if mutation was not applied
    due to inability to do this, not the probability  """
    adapter = DirectAdapter()
    params = get_mutation_params([MutationTypesEnum.reduce], mutation_prob=1)
    mutation = Mutation(**params)

    ind = Individual(adapter.adapt(graph_with_single_node()))
    new_ind = mutation(ind)
    assert new_ind == []

    population = [ind, ind]
    new_population = mutation(population)
    assert new_population == []
