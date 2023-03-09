from datetime import timedelta
from functools import partial
from typing import Type, Optional, Sequence

import networkx as nx

from examples.synthetic_graph_evolution.experiment import run_experiments
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import size_diff


def tree_search_setup(target_graph: nx.DiGraph,
                      optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                      node_types: Sequence[str] = ('X',),
                      timeout: Optional[timedelta] = None,
                      num_iterations: Optional[int] = None):
    # Setup parameters
    num_nodes = target_graph.number_of_nodes()
    requirements = GraphRequirements(
        max_arity=num_nodes,
        max_depth=num_nodes,
        early_stopping_timeout=5,
        early_stopping_iterations=1000,
        keep_n_best=4,
        max_graph_fit_time=timedelta(seconds=15),
        timeout=timeout,
        num_of_generations=num_iterations,
        n_jobs=1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=True,
        mutation_types=[
            MutationTypesEnum.tree_growth,
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
            MutationTypesEnum.single_edge,
        ],
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=DEFAULT_DAG_RULES,
        available_node_types=node_types,
    )

    # Setup objective
    objective = Objective(
        quality_metrics={'edit_dist': partial(tree_edit_dist, target_graph)},
        complexity_metrics={'graph_size': partial(size_diff, target_graph)},
        is_multi_objective=True
    )

    # Generate simple initial population with tree graphs
    initial_graphs = [generate_labeled_graph('tree', k+1)
                      for k in range(gp_params.pop_size)]
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


if __name__ == '__main__':
    results_log = run_experiments(optimizer_setup=tree_search_setup,
                                  optimizer_cls=EvoGraphOptimizer,
                                  graph_names=['tree'],
                                  graph_sizes=[32],
                                  num_trials=1,
                                  trial_timeout=5,
                                  trial_iterations=2000,
                                  visualize=True)
    print(results_log)
