from datetime import timedelta
from functools import partial
from typing import Optional, Sequence, Callable, List
import networkx as nx

from examples.adaptive_optimizer.experiment_setup import run_adaptive_mutations
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.generators import postprocess_nx_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import spectral_dist, size_diff, degree_distance


def generate_gnp_graphs(graph_size: int,
                        gnp_probs: Sequence[float] = (0.05, 0.1, 0.15, 0.3),
                        node_types: Sequence[str] = ('x',)
                        ):
    targets = []
    for prob in gnp_probs:
        # Generate target graph
        nx_graph = nx.gnp_random_graph(graph_size, prob, directed=True)
        target_graph = postprocess_nx_graph(nx_graph, node_labels=node_types)
        targets.append(target_graph)
    return targets


def generate_trees(graph_sizes: Sequence[int], node_types: Sequence[str] = ('x',)):
    trees = [nx.random_tree(n, create_using=nx.DiGraph) for n in graph_sizes]
    trees = [postprocess_nx_graph(g, node_labels=node_types) for g in trees]
    return trees


def get_graph_gp_params(objective: Objective, adaptive_mutation_type: MutationAgentTypeEnum,
                        context_agent_type: ContextAgentTypeEnum = None,
                        mutation_types: List[MutationTypesEnum] = None, pop_size: int = None,
                        decaying_factor: float = 1.0, window_size: int = 5):
    mutation_types = mutation_types or [MutationTypesEnum.single_change,
                                        MutationTypesEnum.single_add,
                                        MutationTypesEnum.single_drop]
    return GPAlgorithmParameters(
        adaptive_mutation_type=adaptive_mutation_type,
        context_agent_type=context_agent_type,
        pop_size=pop_size or 21,
        multi_objective=objective.is_multi_objective,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=mutation_types,
        crossover_types=[CrossoverTypesEnum.none],
        decaying_factor=decaying_factor,
        window_size=window_size
    )


def run_experiment_node_num(adaptive_mutation_type: MutationAgentTypeEnum,
                            context_agent_type: Optional[ContextAgentTypeEnum] = None,
                            target_sizes: Sequence[int] = (50, 400),
                            trial_timeout: int = 15,
                            run_func: Callable = run_adaptive_mutations):
    for target_size in target_sizes:
        # Setup simple objective that searches for required graph size (number of nodes)
        objective = Objective({'graph_size': lambda graph: abs(target_size -
                                                               graph.number_of_nodes())})

        # Build the optimizer
        optimizer, _ = graph_search_setup(
            objective=objective,
            optimizer_cls=EvoGraphOptimizer,
            algorithm_parameters=get_graph_gp_params(objective=objective,
                                                     adaptive_mutation_type=adaptive_mutation_type,
                                                     context_agent_type=context_agent_type),
            timeout=timedelta(minutes=trial_timeout),
            num_iterations=target_size * 3
        )
        run_func(optimizer, objective, visualize=True)


def run_experiment_edge_num(adaptive_mutation_type: MutationAgentTypeEnum,
                            context_agent_type: Optional[ContextAgentTypeEnum] = None,
                            target_sizes: Sequence[int] = (100, 400),
                            trial_timeout: int = 15,
                            run_func: Callable = run_adaptive_mutations):
    for target_size in target_sizes:
        # Setup simple objective that searches for required graph size (number of nodes)
        objective = Objective({'graph_size': lambda graph: abs(target_size -
                                                               graph.number_of_edges())})

        # Build the optimizer
        optimizer, _ = graph_search_setup(
            objective=objective,
            optimizer_cls=EvoGraphOptimizer,
            algorithm_parameters=get_graph_gp_params(objective=objective,
                                                     adaptive_mutation_type=adaptive_mutation_type,
                                                     context_agent_type=context_agent_type),
            timeout=timedelta(minutes=trial_timeout),
            num_iterations=target_size * 3,
        )
        run_func(optimizer, objective, visualize=True)


def run_experiment_graphs_ratio_edges_nodes(adaptive_mutation_type: MutationAgentTypeEnum,
                                            context_agent_type: Optional[ContextAgentTypeEnum] = None,
                                            trial_timeout: int = 15,
                                            trial_iterations: Optional[int] = 500,
                                            run_func: Callable = run_adaptive_mutations):
    """In this experiment setup we generate different graphs with different ratios of #Edges/#Nodes.
    Respectively, probabilities of adding edges and adding nodes must be different for different targets."""

    node_types = ['x']
    for target in generate_gnp_graphs(gnp_probs=[0.15, 0.3], graph_size=100, node_types=node_types):
        # Setup objective that measures some graph-theoretic similarity measure
        objective = Objective(
            quality_metrics={
                'sp_adj': partial(spectral_dist, target, kind='adjacency'),
                'sp_lapl': partial(spectral_dist, target, kind='laplacian'),
            },
            complexity_metrics={
                'graph_size': partial(size_diff, target),
                'degree': partial(degree_distance, target),
            },
            is_multi_objective=True,
        )

        # Build the optimizer
        optimizer, _ = graph_search_setup(
            objective=objective,
            optimizer_cls=EvoGraphOptimizer,
            algorithm_parameters=get_graph_gp_params(objective=objective,
                                                     adaptive_mutation_type=adaptive_mutation_type,
                                                     context_agent_type=context_agent_type),
            node_types=node_types,
            timeout=timedelta(minutes=trial_timeout),
            num_iterations=trial_iterations,
        )

        run_func(optimizer, objective, target, visualize=True)


def run_experiment_trees(adaptive_mutation_type: MutationAgentTypeEnum,
                         context_agent_type: Optional[ContextAgentTypeEnum] = None,
                         trial_timeout: int = 15,
                         trial_iterations: Optional[int] = 500,
                         run_func: Callable = run_adaptive_mutations):
    node_types = ['x']
    for target in generate_trees(graph_sizes=[20, 30, 50], node_types=node_types):
        # Setup objective that measures some graph-theoretic similarity measure
        objective = Objective(
            quality_metrics={'edit_dist': partial(tree_edit_dist, target)},
            complexity_metrics={'degree': partial(degree_distance, target)},
            is_multi_objective=False,
        )

        gp_params = GPAlgorithmParameters(
            multi_objective=objective.is_multi_objective,
            mutation_types=[
                MutationTypesEnum.single_add,
                MutationTypesEnum.single_drop,
            ],
            crossover_types=[CrossoverTypesEnum.none],
            adaptive_mutation_type=adaptive_mutation_type,
            context_agent_type=context_agent_type
        )

        # Build the optimizer
        optimizer, _ = tree_search_setup(
            objective=objective,
            optimizer_cls=EvoGraphOptimizer,
            algorithm_parameters=gp_params,
            node_types=node_types,
            timeout=timedelta(minutes=trial_timeout),
            num_iterations=trial_iterations,
        )

        run_func(optimizer, objective, target, visualize=True)


if __name__ == '__main__':
    """Run adaptive optimizer on different targets to see how adaptive agent converges
    to different probabilities of actions (i.e. mutations) for different targets."""
    adaptive_mutation_type = MutationAgentTypeEnum.bandit

    run_experiment_node_num(trial_timeout=2, adaptive_mutation_type=adaptive_mutation_type)
    run_experiment_edge_num(trial_timeout=2, adaptive_mutation_type=adaptive_mutation_type)
    run_experiment_trees(trial_timeout=10, trial_iterations=2000, adaptive_mutation_type=adaptive_mutation_type)
    run_experiment_graphs_ratio_edges_nodes(trial_timeout=10, trial_iterations=2000,
                                            adaptive_mutation_type=adaptive_mutation_type)
