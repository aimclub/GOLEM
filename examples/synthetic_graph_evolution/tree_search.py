from datetime import timedelta
from functools import partial
from typing import Type, Optional, Sequence

import networkx as nx

from examples.synthetic_graph_evolution.experiment_setup import run_experiments
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, AlgorithmParameters
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import degree_distance


def tree_search_setup(target_graph: Optional[nx.DiGraph] = None,
                      objective: Optional[Objective] = None,
                      optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                      algorithm_parameters: Optional[AlgorithmParameters] = None,
                      node_types: Sequence[str] = ('x',),
                      timeout: Optional[timedelta] = None,
                      num_iterations: Optional[int] = None):
    if target_graph is not None and objective is not None:
        raise ValueError('Please provide either target or objective, not both')
    elif target_graph is not None:
        # Setup objective:
        # - primary metric is edit distance between 2 trees
        # - secondary metric is difference in node degree distribution
        objective = Objective(
            quality_metrics={'edit_dist': partial(tree_edit_dist, target_graph)},
            complexity_metrics={'degree': partial(degree_distance, target_graph)},
            is_multi_objective=False,
        )
        max_graph_size = target_graph.number_of_nodes()
    elif objective is not None:
        max_graph_size = 1000
    else:
        raise ValueError()

    # Setup parameters
    requirements = GraphRequirements(
        max_arity=max_graph_size,
        max_depth=max_graph_size,
        early_stopping_timeout=10,
        early_stopping_iterations=num_iterations // 3 if num_iterations else None,
        keep_n_best=4,
        max_graph_fit_time=timedelta(seconds=10),
        timeout=timeout,
        num_of_generations=num_iterations,
        n_jobs=-1,
        history_dir=None,
    )
    default_gp_params = GPAlgorithmParameters(
        multi_objective=objective.is_multi_objective,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
        ],
        crossover_types=[CrossoverTypesEnum.none]
    )
    gp_params = algorithm_parameters or default_gp_params
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=DEFAULT_DAG_RULES,
        available_node_types=node_types,
    )

    # Generate simple initial population with small tree graphs
    initial_graphs = [generate_labeled_graph('tree', 5, node_types)
                      for _ in range(gp_params.pop_size)]
    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


if __name__ == '__main__':
    """
    In this experiment Optimizer is expected to find the target graph
    (exact or almost exact, depending on available time and graph complexity)
    using Tree Edit Distance metric and a random tree (nx.random_tree) as target.

    This convergence can be seen from achieved metrics and visually from graph plots.
    """
    results_log = run_experiments(optimizer_setup=tree_search_setup,
                                  optimizer_cls=EvoGraphOptimizer,
                                  graph_names=['tree'],
                                  graph_sizes=[16],
                                  num_trials=1,
                                  trial_timeout=5,
                                  trial_iterations=100,
                                  visualize=True)
    print(results_log)
