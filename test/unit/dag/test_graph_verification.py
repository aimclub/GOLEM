from functools import partial

import networkx as nx
import pytest
from matplotlib import pyplot as plt

from examples.synthetic_graph_evolution.generators import postprocess_nx_graph, relabel_nx_graph
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_isolated_nodes, ERROR_PREFIX, \
    has_no_self_cycled_nodes, has_no_isolated_components, DEFAULT_DAG_RULES
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.graph_builder import GraphBuilder
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import spectral_dist, size_diff, degree_distance
from test.unit.mocks.common_mocks import MockNode, MockDomainStructure
from test.unit.utils import graph_first


def graph_with_cycle():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    third = MockNode('c', nodes_from=[second, first])
    second.nodes_from.append(third)
    graph = MockDomainStructure([third])
    return graph


def graph_with_isolated_nodes():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    third = MockNode('c', nodes_from=[second])
    isolated = MockNode('d', nodes_from=[])
    graph = MockDomainStructure([third, isolated])
    return graph


def graph_with_cycled_node():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    second.nodes_from.append(second)
    graph = MockDomainStructure([first, second])
    return graph


def graph_with_isolated_components():
    first = MockNode('a')
    second = MockNode('b', nodes_from=[first])
    third = MockNode('c', nodes_from=[])
    fourth = MockNode('d', nodes_from=[third])
    graph = MockDomainStructure([second, fourth])
    return graph


def test_graph_with_cycle_raise_exception():
    graph = graph_with_cycle()
    with pytest.raises(Exception) as exc:
        assert has_no_cycle(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has cycles'


def test_graph_without_cycles_correct():
    graph = graph_first()

    assert has_no_cycle(graph)


def test_graph_with_isolated_nodes_raise_exception():
    graph = graph_with_isolated_nodes()
    with pytest.raises(ValueError) as exc:
        assert has_no_isolated_nodes(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has isolated nodes'


def test_graph_with_self_cycled_nodes_raise_exception():
    graph = graph_with_cycled_node()
    with pytest.raises(Exception) as exc:
        assert has_no_self_cycled_nodes(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has self-cycled nodes'


def test_graph_with_isolated_components_raise_exception():
    graph = graph_with_isolated_components()
    with pytest.raises(Exception) as exc:
        assert has_no_isolated_components(graph)
    assert str(exc.value) == f'{ERROR_PREFIX} Graph has isolated components'


def test_cycled_graphs_evolution():
    target_graph = nx.DiGraph()
    target_graph.add_nodes_from(range(18))
    target_graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3),
                                 (3, 4), (4, 5), (5, 3), (5, 6),
                                 (6, 7), (7, 8), (8, 6), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14),
                                 (14, 15), (15, 16), (16, 11), (16, 17)])
    target_graph = relabel_nx_graph(target_graph, available_names=('x',))
    num_iterations = 20
    objective = Objective(partial(tree_edit_dist, target_graph))
    max_graph_size = target_graph.number_of_nodes()

    # Setup parameters
    requirements = GraphRequirements(
        num_of_generations=num_iterations,
        n_jobs=-1,
        history_dir=None
    )
    gp_params = GPAlgorithmParameters(
        adaptive_mutation_type=MutationAgentTypeEnum.random,
        pop_size=100,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=[
            MutationTypesEnum.single_edge,
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
            MutationTypesEnum.simple,
            MutationTypesEnum.single_change
        ],
        crossover_types=[CrossoverTypesEnum.none]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_isolated_components, has_no_self_cycled_nodes, has_no_isolated_nodes],
        available_node_types=['x'],
    )

    # Generate simple initial population with cyclic graphs
    initial_graphs = [nxgraph_with_cycle(i) for i in range(4, 10)]
    # Build the optimizer
    optimiser = EvoGraphOptimizer(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    found_graphs = optimiser.optimise(objective)
    # Restore the NetworkX graph back from internal Graph representation
    found_graph = graph_gen_params.adapter.restore(found_graphs[0])
    draw_graphs_subplots(target_graph, found_graph, titles=['Target Graph', 'Found Graph'])
    optimiser.history.show.fitness_line()


def nxgraph_with_cycle(nodes_num):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(nodes_num))
    graph.add_edges_from([(i, (i + 1) % (nodes_num-1)) for i in range(nodes_num-1)])
    graph.add_edge(nodes_num-2, nodes_num-1)
    graph = relabel_nx_graph(graph, available_names=('x',))
    nx.draw_networkx(graph)
    plt.show()
    return graph
