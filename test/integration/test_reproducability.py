from functools import partial

import numpy as np
import pytest

from examples.synthetic_graph_evolution.experiment_setup import run_trial
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.utilities.random import set_random_seed


def setup_gp_params(adaptivity_type: MutationAgentTypeEnum):
    default_gp_params = GPAlgorithmParameters(
        multi_objective=False,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
        ],
        crossover_types=[CrossoverTypesEnum.none],
        adaptive_mutation_type=adaptivity_type
    )
    return default_gp_params


@pytest.mark.parametrize('adaptivity_type', MutationAgentTypeEnum)
def test_same_results(adaptivity_type):
    set_random_seed(42)
    target_graph = generate_labeled_graph('tree', 4, node_labels=['x'])
    num_iterations = 30
    gp_params = setup_gp_params(adaptivity_type)
    old_found_graphs, old_history = None, None
    for _ in range(4):
        found_graph, history = run_trial(target_graph=target_graph,
                                         optimizer_setup=partial(tree_search_setup, algorithm_parameters=gp_params),
                                         num_iterations=num_iterations)
        if old_history:
            assert np.allclose(old_history.all_historical_fitness, history.all_historical_fitness)
