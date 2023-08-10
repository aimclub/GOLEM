from datetime import timedelta
from functools import partial
import random

from typing import List, Any

from examples.adaptive_optimizer.mab_experiment_different_targets import get_graph_gp_params
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from experiments.mab.experiment_with_fickle_mutations.fickle_mutations import fake_add_mutation, fake_add_mutation2, \
    fake_add_mutation3
from experiments.mab.mab_synthetic_experiment_helper import MABSyntheticExperimentHelper
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.context_agents import ContextAgentTypeEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective import Objective


def setup_parameters(initial_graphs: List[Graph], bandit_type: MutationAgentTypeEnum,
                     context_agent_type: ContextAgentTypeEnum,
                     target_size: int, trial_timeout: float,
                     mutation_types: List[Any] = None,
                     decaying_factor: float = 1.0,
                     window_size: int = 5):
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
        initial_graphs=initial_graphs,
        node_types=['x', 'y', 'z', 'w', 'v']
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
    timeout = 3.5
    launch_num = 3
    target_size = 200

    # `bandits_to_compare`, `context_agent_types` and `bandit_labels` correlate one to one.
    # Context must be specified for each bandit: for contextual and neural bandits real context must be specified,
    # for simple bandits -- ContextAgentTypeEnum.none
    bandits_to_compare = [MutationAgentTypeEnum.contextual_bandit, MutationAgentTypeEnum.contextual_bandit]
    context_agent_types = [ContextAgentTypeEnum.nodes_num, ContextAgentTypeEnum.nodes_num]
    bandit_labels = ['bandit_base', 'bandit_3_0.6']
    decaying_factors = [1.0, 0.6]
    window_sizes = [1, 3]
    mutation_types = [
        fake_add_mutation,
        fake_add_mutation2,
        fake_add_mutation3
    ]

    setup_parameters_func = partial(setup_parameters, target_size=target_size, trial_timeout=timeout)
    initial_population_func = partial(initial_population_func,
                                      graph_size=[random.randint(3, 5) for _ in range(20)],
                                      pop_size=20)

    helper = MABSyntheticExperimentHelper(timeout=timeout, launch_num=launch_num, bandits_to_compare=bandits_to_compare,
                                          bandit_labels=bandit_labels, context_agent_types=context_agent_types,
                                          n_clusters=1, is_visualize=False, decaying_factors=decaying_factors,
                                          window_sizes=window_sizes)
    results, actions = helper.compare_bandits(initial_population_func=initial_population_func,
                                              setup_parameters=setup_parameters_func)
    helper.show_boxplots()
    helper.show_fitness_lines()
    helper.show_average_action_probabilities(results, actions)
