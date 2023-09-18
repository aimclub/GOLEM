import os.path
import random
from datetime import timedelta
from functools import partial
from pprint import pprint

import pandas as pd
import seaborn as sns

from typing import List, Callable, Sequence, Optional, Dict, Tuple

from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans

from examples.adaptive_optimizer.mab_experiment_different_targets import get_graph_gp_params
from examples.adaptive_optimizer.utils import plot_action_values
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphOptimizer
from golem.core.paths import project_root
from golem.visualisation.opt_history.fitness_line import MultipleFitnessLines


class MABSyntheticExperimentHelper:
    """ Class to provide synthetic experiments without data to compare MABs. """

    def __init__(self, launch_num: int, timeout: float, bandits_to_compare: List[MutationAgentTypeEnum],
                 context_agent_types: List[ContextAgentTypeEnum], bandit_labels: List[str] = None,
                 path_to_save: str = None, is_visualize: bool = False, n_clusters: Optional[int] = None,
                 decaying_factors: List[float] = None, window_sizes: List[int] = None):
        self.launch_num = launch_num
        self.timeout = timeout
        self.bandits_to_compare = bandits_to_compare
        self.context_agent_types = context_agent_types
        self.bandit_labels = bandit_labels or [bandit.name for bandit in bandits_to_compare]
        self.decaying_factors = decaying_factors or [1.0]*len(bandits_to_compare)
        self.window_sizes = window_sizes or [5]*len(bandits_to_compare)
        self.bandit_metrics = dict.fromkeys(bandit for bandit in self.bandit_labels)
        self.path_to_save = path_to_save or os.path.join(project_root(), 'mab')
        self.is_visualize = is_visualize
        self.histories = dict.fromkeys([bandit for bandit in self.bandit_labels])
        self.cluster = MiniBatchKMeans(n_clusters=n_clusters)

    def compare_bandits(self, setup_parameters: Callable, initial_population_func: Callable = None) \
            -> Tuple[dict, list]:
        results = dict()
        for i in range(self.launch_num):
            initial_graphs = initial_population_func()
            for j, bandit in enumerate(self.bandits_to_compare):
                optimizer, objective = setup_parameters(initial_graphs=initial_graphs, bandit_type=bandit,
                                                        context_agent_type=self.context_agent_types[j],
                                                        decaying_factor=self.decaying_factors[j],
                                                        window_size=self.window_sizes[j])
                agent = optimizer.mutation.agent
                result = self.launch_bandit(bandit_type=bandit, optimizer=optimizer, objective=objective, bandit_num=j)
                if self.bandit_labels[j] not in results.keys():
                    results[self.bandit_labels[j]] = []
                results[self.bandit_labels[j]].append(result)
        if self.is_visualize:
            self.show_average_action_probabilities(show_action_probabilities=results, actions=agent.actions)
        return results, agent.actions

    def launch_bandit(self, bandit_type: MutationAgentTypeEnum, optimizer: GraphOptimizer, objective: Callable,
                      bandit_num: int):

        stats_action_value_log: Dict[int, List[List[float]]] = dict()

        def log_action_values(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
            values = optimizer.mutation.agent.get_action_values(obs=next_pop[0])
            if 0 not in stats_action_value_log.keys():
                stats_action_value_log[0] = []
            stats_action_value_log[0].append(list(values))
            # MAB agent can be saved here -- commented not to clog up the memory
            # optimizer.mutation.agent.save()

        def log_action_values_with_clusters(next_pop: PopulationT, optimizer: EvoGraphOptimizer):
            obs_contexts = optimizer.mutation.agent.get_context(next_pop)
            self.cluster.partial_fit(obs_contexts)
            centers = self.cluster.cluster_centers_
            for i, center in enumerate(centers):
                values = optimizer.mutation.agent.get_action_values(obs=[center])
                if i not in stats_action_value_log.keys():
                    stats_action_value_log[i] = []
                stats_action_value_log[i].append(list(values))

        # set iteration callback
        if bandit_type == MutationAgentTypeEnum.bandit:
            optimizer.set_iteration_callback(log_action_values)
        elif bandit_type in (MutationAgentTypeEnum.contextual_bandit, MutationAgentTypeEnum.neural_bandit):
            optimizer.set_iteration_callback(log_action_values_with_clusters)
        else:
            raise ValueError("No callback function was specified for that bandit type.")

        found_graphs = optimizer.optimise(objective)
        found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs

        history = optimizer.history
        bandit_label = self.bandit_labels[bandit_num]
        if not self.histories[bandit_label]:
            self.histories[bandit_label] = []
        self.histories[bandit_label].append(history)

        agent = optimizer.mutation.agent
        found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
        final_metrics = objective(found_nx_graph).value

        if not self.bandit_metrics[bandit_label]:
            self.bandit_metrics[bandit_label] = []
        self.bandit_metrics[bandit_label].append(final_metrics)

        print('History of action probabilities:')
        pprint(stats_action_value_log)
        if self.is_visualize:
            self.show_fitness_line(found_nx_graph=found_nx_graph, final_metrics=final_metrics,
                                   history=history)
            self.show_action_probabilities(bandit_type=bandit_type, stats_action_value_log=stats_action_value_log,
                                           actions=agent.actions)

        return stats_action_value_log

    @staticmethod
    def show_fitness_line(found_nx_graph, final_metrics, history):
        draw_graphs_subplots(found_nx_graph, titles=[f'Found Graph (fitness={final_metrics})'])
        history.show.fitness_line()

    def show_action_probabilities(self, bandit_type: MutationAgentTypeEnum, stats_action_value_log,
                                  actions, is_average: bool = False):
        if is_average:
            titles = ['Average action Expectation Values', 'Average action Probabilities']
        else:
            titles = ['Action Expectation Values', 'Action Probabilities']
        if bandit_type == MutationAgentTypeEnum.bandit:
            plot_action_values(stats=stats_action_value_log[0], action_tags=actions, titles=titles)
            plt.show()
        else:
            centers = self.cluster.cluster_centers_
            for i in range(self.cluster.n_clusters):
                if len(centers[i]) > 1:
                    titles_centers = [title + f' for cluster with center idx={i}' for title in titles]
                else:
                    titles_centers = [title + f' for cluster with center {centers[i]}' for title in titles]
                plot_action_values(stats=stats_action_value_log[i], action_tags=actions,
                                   titles=titles_centers)
                plt.show()

    def show_average_action_probabilities(self, show_action_probabilities: dict, actions):
        """ Shows action probabilities across several launches. """
        for idx, bandit in enumerate(list(show_action_probabilities.keys())):
            total_sum = None
            for launch in show_action_probabilities[bandit]:
                if not total_sum:
                    total_sum = launch
                    break
                for cluster in launch.keys():
                    for action_probabilities_list_idx in range(len(total_sum[cluster])):
                        for action_probability_idx in range(len(total_sum[cluster][action_probabilities_list_idx])):
                            if action_probabilities_list_idx >= len(launch[cluster]):
                                continue
                            if action_probability_idx >= len(launch[cluster][action_probabilities_list_idx]):
                                continue
                            total_sum[cluster][action_probabilities_list_idx][action_probability_idx] += \
                                launch[cluster][action_probabilities_list_idx][action_probability_idx]
            for cluster in total_sum.keys():
                for action_probabilities_list_idx in range(len(total_sum[cluster])):
                    for action_probability_idx in range(len(total_sum[cluster][action_probabilities_list_idx])):
                        total_sum[cluster][action_probabilities_list_idx][action_probability_idx] /= \
                            len(show_action_probabilities[bandit])
            self.show_action_probabilities(bandit_type=MutationAgentTypeEnum(self.bandits_to_compare[idx]),
                                           stats_action_value_log=total_sum,
                                           actions=actions,
                                           is_average=True)

    def show_boxplots(self):
        sns.boxplot(data=pd.DataFrame(self.bandit_metrics))
        plt.title('Metrics', fontsize=15)
        plt.show()

    def show_fitness_lines(self):
        multiple_fitness_lines = MultipleFitnessLines(histories_to_compare=self.histories)
        multiple_fitness_lines.visualize()


def setup_parameters(initial_graphs: List[Graph], bandit_type: MutationAgentTypeEnum,
                     context_agent_type: ContextAgentTypeEnum,
                     target_size: int, trial_timeout: float,
                     mutation_types: List[MutationTypesEnum] = None,
                     decaying_factor: float = 1.0,
                     window_size: int = 1):
    objective = Objective({'graph_size': lambda graph: abs(target_size -
                                                           graph.number_of_nodes())})

    # Build the optimizer
    optimizer, _ = graph_search_setup(
        objective=objective,
        optimizer_cls=EvoGraphOptimizer,
        algorithm_parameters=get_graph_gp_params(objective=objective,
                                                 adaptive_mutation_type=bandit_type,
                                                 context_agent_type=context_agent_type,
                                                 decaying_factor=decaying_factor,
                                                 window_size=window_size,
                                                 mutation_types=mutation_types),
        timeout=timedelta(minutes=trial_timeout),
        num_iterations=target_size * 3,
        initial_graphs=initial_graphs
    )
    return optimizer, objective


def initial_population_func(graph_size: List[int] = None, pop_size: int = None, initial_graphs: List[Graph] = None):
    if initial_graphs:
        return initial_graphs
    initial_graphs = [generate_labeled_graph('tree', graph_size[i], directed=True, node_labels=['x'])
                      for i in range(pop_size)]
    initial_opt_graphs = [BaseNetworkxAdapter().adapt(item=graph) for graph in initial_graphs]
    return initial_opt_graphs


if __name__ == '__main__':
    timeout = 0.3
    launch_num = 1
    target_size = 50

    # `bandits_to_compare`, `context_agent_types` and `bandit_labels` correlate one to one.
    # Context must be specified for each bandit: for contextual and neural bandits real context must be specified,
    # for simple bandits -- ContextAgentTypeEnum.none
    bandits_to_compare = [MutationAgentTypeEnum.bandit, MutationAgentTypeEnum.contextual_bandit]
    context_agent_types = [ContextAgentTypeEnum.none_encoding, ContextAgentTypeEnum.operations_quantity]
    bandit_labels = ['simple_bandit', f'context_{context_agent_types[1].name}']

    setup_parameters_func = partial(setup_parameters, target_size=target_size, trial_timeout=timeout)
    initial_population_func = partial(initial_population_func,
                                      graph_size=[random.randint(5, 10) for _ in range(10)] +
                                                 [random.randint(90, 95) for _ in range(10)],
                                      pop_size=20)

    helper = MABSyntheticExperimentHelper(timeout=timeout, launch_num=launch_num, bandits_to_compare=bandits_to_compare,
                                          bandit_labels=bandit_labels, context_agent_types=context_agent_types,
                                          n_clusters=2, is_visualize=True)
    helper.compare_bandits(initial_population_func=initial_population_func,
                           setup_parameters=setup_parameters_func)
    helper.show_boxplots()
    helper.show_fitness_lines()
