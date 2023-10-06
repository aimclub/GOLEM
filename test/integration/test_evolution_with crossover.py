from functools import partial

import networkx as nx

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.log import Log
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.metrics.graph_metrics import spectral_dist
from golem.serializers.serializer import default_save


def test_evolution_with_crossover():
    Log().reset_logging_level(10)
    target_graph = generate_labeled_graph('tree', 50).reverse()
    num_iterations = 100
    objective = Objective(partial(spectral_dist, target_graph))

    requirements = GraphRequirements(
        early_stopping_iterations=num_iterations,
        num_of_generations=num_iterations,
        n_jobs=-1,
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
        crossover_types=[CrossoverTypesEnum.one_point]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=DEFAULT_DAG_RULES,
        available_node_types=['x'],
    )

    # Generate simple initial population with cyclic graphs
    initial_graphs = [generate_labeled_graph('tree', i).reverse() for i in range(4, 20)]

    optimiser = EvoGraphOptimizer(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    found_graphs = optimiser.optimise(objective)
    found_graph: nx.DiGraph = graph_gen_params.adapter.restore(found_graphs[0])
    assert found_graph is not None
    assert len(found_graph.nodes) > 0
