from datetime import timedelta
from pprint import pprint
from typing import Callable, Optional, List

import networkx as nx
from matplotlib import pyplot as plt

from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.generators import postprocess_nx_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots, plot_action_values
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.operators.operator import PopulationT
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


def generate_trees(graph_sizes: Sequence[int]):
    trees = [nx.random_tree(n, create_using=nx.DiGraph) for n in graph_sizes]
    trees = [postprocess_nx_graph(g) for g in trees]
    return trees


def run_adaptive_mutations(
        target: nx.DiGraph,
        optimizer_setup: Callable = graph_search_setup,
        trial_timeout: int = 15,
        trial_iterations: Optional[int] = 500,
        visualize: bool = True,
):
    node_types = ['x']
    stats_action_value_log: List[List[float]] = []

    def log_action_values(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
        values = optimizer.mutation.agent.get_action_values(obs=None)
        stats_action_value_log.append(list(values))

    # Build the optimizer and setup the logger
    optimizer, objective = optimizer_setup(
        target,
        optimizer_cls=EvoGraphOptimizer,
        node_types=node_types,
        timeout=timedelta(minutes=trial_timeout),
        num_iterations=trial_iterations,
    )
    optimizer.set_iteration_callback(log_action_values)
    # Run the optimizer
    found_graphs = optimizer.optimise(objective)
    found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
    history = optimizer.history
    agent = optimizer.mutation.agent

    print('History of action probabilities:')
    pprint(stats_action_value_log)
    if visualize:
        found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
        draw_graphs_subplots(target, found_nx_graph,
                             titles=['Target Graph', 'Found Graph'])
        history.show.fitness_line()
        plot_action_values(stats_action_value_log, action_tags=agent.actions)
        plt.show()
    return stats_action_value_log


if __name__ == '__main__':
    """Run adaptive optimizer on different targets to see how adaptive agent converges 
    to different probabilities of actions (i.e. mutations) for different targets."""

    target_graphs = generate_gnp_graphs(gnp_probs=[0.15, 0.3], graph_size=100)
    for target in target_graphs:
        run_adaptive_mutations(target,
                               optimizer_setup=graph_search_setup,
                               trial_iterations=200,
                               trial_timeout=30,
                               visualize=True)

    trees = generate_trees(graph_sizes=[10, 20, 30, 50])
    for tree in trees:
        run_adaptive_mutations(tree,
                               optimizer_setup=tree_search_setup,
                               trial_iterations=2000,
                               trial_timeout=30,
                               visualize=True)
