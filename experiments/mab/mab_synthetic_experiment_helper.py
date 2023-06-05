import os.path
import random
from datetime import timedelta
from functools import partial
from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns

from typing import List, Callable, Sequence, Optional, Dict

import networkx as nx
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans

from examples.adaptive_optimizer.mab_experiment_different_targets import get_graph_gp_params
from examples.adaptive_optimizer.utils import plot_action_values
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphOptimizer
from golem.core.paths import project_root
from golem.visualisation.opt_history.fitness_line import MultipleFitnessLines


class MABSyntheticExperimentHelper:
    """ Class to provide synthetic experiments without data to compare MABs. """
    def __init__(self, launch_num: int, timeout: float, bandits_to_compare: List[MutationAgentTypeEnum],
                 path_to_save: str = None, is_visualize: bool = False, n_clusters: Optional[int] = None):
        self.launch_num = launch_num
        self.timeout = timeout
        self.bandits_to_compare = bandits_to_compare
        self.bandit_metrics = dict.fromkeys(bandit.name for bandit in self.bandits_to_compare)
        self.path_to_save = path_to_save or os.path.join(project_root(), 'mab')
        self.is_visualize = is_visualize
        self.histories = dict.fromkeys([bandit.name for bandit in self.bandits_to_compare])
        self.cluster = MiniBatchKMeans(n_clusters=n_clusters)

    def compare(self, setup_parameters: Callable, initial_population_func: Callable = None):
        for i in range(self.launch_num):
            initial_graphs = initial_population_func()
            for bandit in self.bandits_to_compare:
                optimizer, objective = setup_parameters(initial_graphs=initial_graphs, bandit_type=bandit)
                self.launch_bandit(bandit_type=bandit, optimizer=optimizer, objective=objective)

    def launch_bandit(self, bandit_type: MutationAgentTypeEnum, optimizer: GraphOptimizer, objective: Callable):

        stats_action_value_log: Dict[int, List[List[float]]] = dict()

        def log_action_values(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
            values = optimizer.mutation.agent.get_action_values(obs=next_pop[0])
            if 0 not in stats_action_value_log.keys():
                stats_action_value_log[0] = []
            stats_action_value_log[0].append(list(values))

        def log_action_values_with_clusters(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
            obs_contexts = optimizer.mutation.agent.get_context(next_pop)
            self.cluster.partial_fit(np.array(obs_contexts).reshape(-1, 1))
            centers = self.cluster.cluster_centers_
            for i, center in enumerate(sorted(centers)):
                values = optimizer.mutation.agent.get_action_values(obs=[center])
                if i not in stats_action_value_log.keys():
                    stats_action_value_log[i] = []
                stats_action_value_log[i].append(list(values))

        # set iteration callback
        if bandit_type == MutationAgentTypeEnum.bandit:
            optimizer.set_iteration_callback(log_action_values)
        else:
            optimizer.set_iteration_callback(log_action_values_with_clusters)

        found_graphs = optimizer.optimise(objective)
        found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
        history = optimizer.history
        if not self.histories[bandit_type.name]:
            self.histories[bandit_type.name] = []
        self.histories[bandit_type.name].append(history)
        agent = optimizer.mutation.agent
        found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
        final_metrics = objective(found_nx_graph).value
        if not self.bandit_metrics[bandit_type.name]:
            self.bandit_metrics[bandit_type.name] = []
        self.bandit_metrics[bandit_type.name].append(final_metrics)

        print('History of action probabilities:')
        pprint(stats_action_value_log)
        if self.is_visualize:
            self.show_fitness_line(found_nx_graph=found_nx_graph, final_metrics=final_metrics,
                                   history=history)
            self.show_action_probabilities(bandit_type=bandit_type, stats_action_value_log=stats_action_value_log,
                                           agent=agent)

    @staticmethod
    def show_fitness_line(found_nx_graph, final_metrics, history):
        draw_graphs_subplots(found_nx_graph, titles=[f'Found Graph (fitness={final_metrics})'])
        history.show.fitness_line()

    def show_action_probabilities(self, bandit_type: MutationAgentTypeEnum, stats_action_value_log, agent):
        if bandit_type == MutationAgentTypeEnum.bandit:
            plot_action_values(stats_action_value_log[0], action_tags=agent.actions)
            plt.show()
        else:
            for i in range(self.cluster.n_clusters):
                plot_action_values(stats_action_value_log[i], action_tags=agent.actions,
                                   cluster_center=f'{self.cluster.cluster_centers_[i]}')
                plt.show()

    def show_boxplots(self):
        sns.boxplot(data=pd.DataFrame(self.bandit_metrics))
        plt.title(f'Metrics', fontsize=15)
        plt.show()

    def show_fitness_lines(self):
        multiple_fitness_lines = MultipleFitnessLines(histories_to_compare=self.histories)
        multiple_fitness_lines.visualize()


def setup_parameters(initial_graphs: List[Graph], bandit_type: MutationAgentTypeEnum,
                     target_size: int, trial_timeout: float):
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


def initial_population_func(graph_size: List[int] = None, pop_size: int = None, initial_graphs: List[Graph] = None):
    if initial_graphs:
        return initial_graphs
    initial_graphs = [nx.random_tree(graph_size[i], create_using=nx.DiGraph)
                      for i in range(pop_size)]
    return initial_graphs


if __name__ == '__main__':
    timeout = 0.5
    launch_num = 1
    target_size = 50

    bandits_to_compare = [MutationAgentTypeEnum.bandit, MutationAgentTypeEnum.contextual_bandit]
    setup_parameters_func = partial(setup_parameters, target_size=target_size, trial_timeout=timeout)
    initial_population_func = partial(initial_population_func,
                                      # graph_size=[random.randint(5, 7) for _ in range(21)],
                                      graph_size=[random.randint(5, 10) for _ in range(19)] +
                                                 [random.randint(90, 95) for _ in range(2)],
                                      pop_size=21)

    helper = MABSyntheticExperimentHelper(timeout=timeout, launch_num=launch_num, bandits_to_compare=bandits_to_compare,
                                          n_clusters=2, is_visualize=True)
    helper.compare(initial_population_func=initial_population_func,
                   setup_parameters=setup_parameters_func)
    # helper.show_boxplots()
    # helper.show_fitness_lines()
