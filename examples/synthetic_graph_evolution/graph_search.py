from datetime import timedelta
from functools import partial
from typing import Type, Optional

from examples.synthetic_graph_evolution.experiment import run_experiments
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.optimisers.adaptive.operatoragent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.metrics.graph_metrics import *


def graph_search_setup(target_graph: nx.DiGraph,
                       optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                       node_types: Sequence[str] = ('x',),
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
        timeout=timeout,
        num_of_generations=num_iterations,
        n_jobs=-1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        adaptive_mutation_type=MutationAgentTypeEnum.random,
        pop_size=21,
        multi_objective=False,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_edge,
            MutationTypesEnum.single_drop,
        ],
        crossover_types=[CrossoverTypesEnum.none]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes],
        available_node_types=node_types,
    )

    # Setup objective that measures some graph-theoretic similarity measure
    objective = Objective(
        quality_metrics={
            # 'sp_adj': partial(spectral_dist, target_graph, kind='adjacency'),
            # 'sp_lapl': partial(spectral_dist, target_graph, kind='laplacian'),
            'degree': partial(degree_distance, target_graph),
        },
        complexity_metrics={
            'graph_size': partial(size_diff, target_graph),
        },
        is_multi_objective=gp_params.multi_objective,
    )

    ### Edge to node ratio
    # def edge2node_ratio(graph) -> float:
    #     return - graph.number_of_edges() / graph.number_of_nodes()
    #
    # target = edge2node_ratio(target_graph)
    #
    # objective = Objective({'edge/node ratio':
    #                            edge2node_ratio
    #                            # lambda graph: abs(target - edge2node_ratio(graph))
    #                        })

    ### Graph size
    # objective = Objective({'graph_size': lambda graph: -graph.number_of_nodes()})
    objective = Objective({'graph_size': lambda graph: abs(target_graph.number_of_nodes() -
                                                           graph.number_of_nodes())})

    # Generate simple initial population with line graphs
    initial_graphs = [generate_labeled_graph('line', k+3, node_types)
                      for k in range(gp_params.pop_size)]
    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


if __name__ == '__main__':
    results_log = run_experiments(optimizer_setup=graph_search_setup,
                                  optimizer_cls=EvoGraphOptimizer,
                                  graph_names=['gnp'],
                                  graph_sizes=[50],
                                  num_trials=1,
                                  trial_timeout=15,
                                  trial_iterations=1000,
                                  visualize=True)
    print(results_log)
