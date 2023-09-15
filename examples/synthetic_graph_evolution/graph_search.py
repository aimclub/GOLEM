import os.path
from datetime import timedelta
from functools import partial
from typing import Type, Optional, Sequence, List

import networkx as nx
from examples.synthetic_graph_evolution.experiment_setup import run_experiments
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer, AlgorithmParameters
from golem.core.paths import project_root
from golem.metrics.graph_metrics import spectral_dist, size_diff, degree_distance


def graph_search_setup(target_graph: Optional[nx.DiGraph] = None,
                       objective: Optional[Objective] = None,
                       optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                       adaptive_mutation_type: Optional[MutationAgentTypeEnum] = None,
                       context_agent_type: Optional[ContextAgentTypeEnum] = None,
                       algorithm_parameters: Optional[AlgorithmParameters] = None,
                       node_types: Sequence[str] = ('x',),
                       timeout: Optional[timedelta] = None,
                       num_iterations: Optional[int] = None,
                       initial_graph_sizes: Optional[List[int]] = None,
                       initial_graphs: List[Graph] = None,
                       graph_name: str = None,
                       pop_size: int = None,
                       path_to_save_agent: str = None):
    if target_graph is not None and objective is not None:
        raise ValueError('Please provide either target or objective, not both')
    elif target_graph is not None:
        # Setup objective that measures some graph-theoretic similarity measure
        objective = Objective(
            quality_metrics={
                'sp_adj': partial(spectral_dist, target_graph, kind='adjacency'),
                'sp_lapl': partial(spectral_dist, target_graph, kind='laplacian'),
            },
            complexity_metrics={
                'graph_size': partial(size_diff, target_graph),
                'degree': partial(degree_distance, target_graph),
            },
            is_multi_objective=True
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
        early_stopping_iterations=3,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        n_jobs=1,
        history_dir=None,
        agent_dir=path_to_save_agent
    )
    adaptive_mutation_type = adaptive_mutation_type or MutationAgentTypeEnum.random
    default_gp_params = GPAlgorithmParameters(
        adaptive_mutation_type=adaptive_mutation_type,
        context_agent_type=context_agent_type,
        pop_size=pop_size or 21,
        multi_objective=objective.is_multi_objective,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_edge,
            MutationTypesEnum.single_drop
        ],
        crossover_types=[CrossoverTypesEnum.none]
    )
    gp_params = algorithm_parameters or default_gp_params
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=DEFAULT_DAG_RULES,
        available_node_types=node_types,
    )

    # # Generate simple initial population with line graphs
    # if not initial_graphs:
    #     if not initial_graph_sizes:
    #         initial_graph_sizes = [7] * gp_params.pop_size
    #     initial_graphs = [nx.random_tree(initial_graph_sizes[i], create_using=nx.DiGraph)
    #                       for i in range(gp_params.pop_size)]
    initial_graphs = [generate_labeled_graph(graph_name, 5, node_types) for _ in range(10)]
    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


if __name__ == '__main__':
    path_to_save = os.path.join(project_root(), 'experiments', 'mab',
                                'experiment_random_golem_vs_mab', 'results')
    graph_search_random = graph_search_setup
    graph_search_bandit = partial(graph_search_setup,
                                  adaptive_mutation_type=MutationAgentTypeEnum.bandit)
    graph_search_context_bandit = partial(graph_search_setup,
                                          adaptive_mutation_type=MutationAgentTypeEnum.contextual_bandit,
                                          context_agent_type=ContextAgentTypeEnum.adjacency_matrix)

    graph_searches = [graph_search_random, graph_search_bandit, graph_search_context_bandit]
    graph_setup_names = ['random', 'bandit', 'context_bandit']

    for i, search in enumerate(graph_searches):
        results_log = run_experiments(optimizer_setup=search,
                                      optimizer_cls=EvoGraphOptimizer,
                                      setup_name=graph_setup_names[i],
                                      graph_names=['gnp', 'tree', 'grid2d'],
                                      graph_sizes=[20, 50, 100],
                                      num_trials=15,
                                      trial_iterations=1000,
                                      visualize=True,
                                      path_to_save=path_to_save)
        print(results_log)
