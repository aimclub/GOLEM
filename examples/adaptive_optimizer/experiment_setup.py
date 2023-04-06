from pprint import pprint
from typing import List, Sequence, Optional

import networkx as nx
from matplotlib import pyplot as plt

from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from examples.adaptive_optimizer.utils import plot_action_values
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.objective import Objective


def run_adaptive_mutations(
        optimizer: EvoGraphOptimizer,
        objective: Objective,
        target: Optional[nx.DiGraph] = None,
        visualize: bool = True,
):
    """This experiment setup outputs graphic of relative action probabilities
    for given target/objective and given optimizer setup."""
    stats_action_value_log: List[List[float]] = []

    def log_action_values(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
        values = optimizer.mutation.agent.get_action_values(obs=None)
        stats_action_value_log.append(list(values))

    # Setup the logger and run the optimizer
    optimizer.set_iteration_callback(log_action_values)
    found_graphs = optimizer.optimise(objective)
    found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
    history = optimizer.history
    agent = optimizer.mutation.agent

    print('History of action probabilities:')
    pprint(stats_action_value_log)
    if visualize:
        found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
        final_metrics = objective(found_nx_graph).value
        if target is not None:
            draw_graphs_subplots(target, found_nx_graph,
                                 titles=['Target Graph', f'Found Graph (fitness={final_metrics})'])
        else:
            draw_graphs_subplots(found_nx_graph, titles=[f'Found Graph (fitness={final_metrics})'])
        history.show.fitness_line()
        plot_action_values(stats_action_value_log, action_tags=agent.actions)
        plt.show()
    return stats_action_value_log
