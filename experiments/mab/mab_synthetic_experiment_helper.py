import os.path
import random
from datetime import timedelta
from functools import partial
from pprint import pprint

from typing import List, Callable, Sequence

from matplotlib import pyplot as plt

from examples.adaptive_optimizer.mab_experiment_different_targets import get_graph_gp_params
from examples.adaptive_optimizer.utils import plot_action_values
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphOptimizer
from golem.core.paths import project_root
from golem.visualisation.opt_history.fitness_line import MultipleFitnessLines


class MABSyntheticExperimentHelper:
    def __init__(self, launch_num: int, timeout: float, bandits_to_compare: List[MutationAgentTypeEnum],
                 path_to_save: str = None, is_visualize: bool = False):
        self.launch_num = launch_num
        self.timeout = timeout
        self.bandits_to_compare = bandits_to_compare
        self.bandit_metrics = dict.fromkeys(self.bandits_to_compare)
        self.path_to_save = path_to_save or os.path.join(project_root(), 'mab')
        self.is_visualize = is_visualize
        self.histories = dict.fromkeys([bandit.name for bandit in self.bandits_to_compare])

    def compare(self, setup_parameters: Callable):
        for i in range(self.launch_num):
            initial_graphs = self._initial_population()
            for bandit in self.bandits_to_compare:
                optimizer, objective = setup_parameters(initial_graphs=initial_graphs, bandit_type=bandit)
                self.launch_bandit(bandit_type=bandit, optimizer=optimizer, objective=objective)

    def launch_bandit(self, bandit_type: MutationAgentTypeEnum, optimizer: GraphOptimizer, objective: Callable):

        stats_action_value_log: List[List[float]] = []

        def log_action_values(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
            values = optimizer.mutation.agent.get_action_values(obs=next_pop[0])
            stats_action_value_log.append(list(values))

        optimizer.set_iteration_callback(log_action_values)
        found_graphs = optimizer.optimise(objective)
        found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
        history = optimizer.history
        if not self.histories[bandit_type.name]:
            self.histories[bandit_type.name] = []
        self.histories[bandit_type.name].append(history)
        agent = optimizer.mutation.agent
        found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
        final_metrics = objective(found_nx_graph).value
        if not self.bandit_metrics[bandit_type]:
            self.bandit_metrics[bandit_type] = []
        self.bandit_metrics[bandit_type].append(final_metrics)

        print('History of action probabilities:')
        pprint(stats_action_value_log)
        if self.is_visualize:
            draw_graphs_subplots(found_nx_graph, titles=[f'Found Graph (fitness={final_metrics})'])
            history.show.fitness_line()
            plot_action_values(stats_action_value_log, action_tags=agent.actions)
            plt.show()

    def _initial_population(self) -> List[OptGraph]:
        graph_size = [random.randint(3, 10) for _ in range(21)]
        node_types = ('x',)
        initial_graphs = [generate_labeled_graph('gnp', graph_size[i], node_types) for i in range(21)]
        return initial_graphs

    def show_boxplots(self):
        plt.boxplot(x=list(self.bandit_metrics.values()))
        plt.xlabel(i.bandit for i in list(self.bandit_metrics.keys()))
        plt.xticks(rotation=45)
        plt.title(f'Metrics', fontsize=15)
        plt.show()

    def show_fitness_lines(self):
        multiple_fitness_lines = MultipleFitnessLines(histories_to_compare=self.histories)
        multiple_fitness_lines.visualize()


def setup_parameters(initial_graphs, bandit_type: MutationAgentTypeEnum):
    target_size = 100
    trial_timeout = 0.5
    objective = Objective({'graph_size': lambda graph: abs(target_size -
                                                           graph.number_of_nodes())})

    # Build the optimizer
    optimizer, _ = graph_search_setup(
        objective=objective,
        optimizer_cls=EvoGraphOptimizer,
        algorithm_parameters=get_graph_gp_params(objective=objective,
                                                 adaptive_mutation_type=bandit_type),
        timeout=timedelta(minutes=trial_timeout),
        num_iterations=target_size * 3,
        initial_graphs=initial_graphs
    )
    return optimizer, objective


if __name__ == '__main__':
    timeout = 3
    launch_num = 5
    bandits_to_compare = [MutationAgentTypeEnum.contextual_bandit, MutationAgentTypeEnum.bandit]
    setup_parameters_func = setup_parameters
    helper = MABSyntheticExperimentHelper(timeout=timeout, launch_num=launch_num, bandits_to_compare=bandits_to_compare)
    helper.compare(setup_parameters=setup_parameters_func)
    helper.show_boxplots()
    helper.show_fitness_lines()
