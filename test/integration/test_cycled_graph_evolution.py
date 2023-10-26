from functools import partial

import networkx as nx

from examples.synthetic_graph_evolution.generators import relabel_nx_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import has_no_isolated_components, has_no_self_cycled_nodes, \
    has_no_isolated_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, subgraph_crossover
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.metrics.graph_metrics import spectral_dist


def nxgraph_with_cycle(nodes_num):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(nodes_num))
    graph.add_edges_from([(i, (i + 1) % (nodes_num - 1)) for i in range(nodes_num - 1)])
    graph.add_edge(nodes_num - 2, nodes_num - 1)
    graph = relabel_nx_graph(graph, available_names=('x',))
    return graph


def test_cycled_graphs_evolution():
    target_graph = nx.DiGraph()
    target_graph.add_nodes_from(range(18))
    target_graph.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3),
                                 (3, 4), (4, 5), (5, 3), (5, 6),
                                 (6, 7), (7, 8), (8, 6), (8, 9),
                                 (9, 10), (10, 11), (11, 12), (12, 13),
                                 (13, 14), (14, 15), (15, 16), (16, 11),
                                 (16, 17)])
    target_graph = relabel_nx_graph(target_graph, available_names=('x',))
    num_iterations = 50
    objective = Objective(partial(spectral_dist, target_graph))

    requirements = GraphRequirements(
        early_stopping_iterations=num_iterations,
        num_of_generations=num_iterations,
        n_jobs=1,
        history_dir=None
    )
    gp_params = GPAlgorithmParameters(
        pop_size=30,
        mutation_types=[
            MutationTypesEnum.single_edge,
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
            MutationTypesEnum.simple,
            MutationTypesEnum.single_change
        ],
        crossover_types=[subgraph_crossover]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_isolated_components, has_no_self_cycled_nodes, has_no_isolated_nodes],
        available_node_types=['x'],
    )

    # Generate simple initial population with cyclic graphs
    initial_graphs = [nxgraph_with_cycle(i) for i in range(4, 20)]

    optimiser = EvoGraphOptimizer(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    found_graphs = optimiser.optimise(objective)
    found_graph: nx.DiGraph = graph_gen_params.adapter.restore(found_graphs[0])
    assert found_graph is not None
    assert len(found_graph.nodes) > 0
