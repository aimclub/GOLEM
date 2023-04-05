from datetime import timedelta
from functools import partial
from typing import Optional

from examples.adaptive_optimizer.experiment_setup import run_adaptive_mutations
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.generators import postprocess_nx_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective import Objective
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import *


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


def run_experiment_graphs(trial_timeout: int = 15, trial_iterations: Optional[int] = 500):
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
            node_types=node_types,
            timeout=timedelta(minutes=trial_timeout),
            num_iterations=trial_iterations,
        )

        run_adaptive_mutations(target, objective, optimizer, visualize=True)


def run_experiment_trees(trial_timeout: int = 15, trial_iterations: Optional[int] = 500):
    node_types = ['x']
    for target in generate_trees(graph_sizes=[10, 20, 30, 50], node_types=node_types):
        # Setup objective that measures some graph-theoretic similarity measure
        objective = Objective(
            quality_metrics={'edit_dist': partial(tree_edit_dist, target)},
            complexity_metrics={'degree': partial(degree_distance, target)},
            is_multi_objective=False,
        )

        # Build the optimizer
        optimizer, _ = tree_search_setup(
            objective=objective,
            optimizer_cls=EvoGraphOptimizer,
            node_types=node_types,
            timeout=timedelta(minutes=trial_timeout),
            num_iterations=trial_iterations,
        )

        run_adaptive_mutations(target, objective, optimizer, visualize=True)


if __name__ == '__main__':
    """Run adaptive optimizer on different targets to see how adaptive agent converges 
    to different probabilities of actions (i.e. mutations) for different targets."""

    run_experiment_trees(trial_timeout=15, trial_iterations=2000)
    run_experiment_graphs(trial_timeout=15, trial_iterations=2000)
