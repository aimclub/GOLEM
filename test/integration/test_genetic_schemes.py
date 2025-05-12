from typing import Sequence

import numpy as np
import pytest

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.utilities.utilities import set_random_seed


def set_up_params(genetic_scheme: GeneticSchemeTypesEnum):
    gp_params = GPAlgorithmParameters(
        multi_objective=False,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop
        ],
        crossover_types=[CrossoverTypesEnum.none],
        genetic_scheme_type=genetic_scheme
    )
    return gp_params


@pytest.mark.parametrize('genetic_type', GeneticSchemeTypesEnum)
def test_genetic_scheme_types(genetic_type):
    set_random_seed(42)
    target_graph = generate_labeled_graph('tree', 30, node_labels=['x'])
    num_iterations = 30

    gp_params = set_up_params(genetic_type)
    optimizer, objective = tree_search_setup(target_graph,
                                             num_iterations=num_iterations,
                                             algorithm_parameters=gp_params)
    found_graphs = optimizer.optimise(objective)
    found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
    history = optimizer.history
    assert found_graph is not None
    # at least 20% more generation than early_stopping_iterations were evaluated
    # (+2 gen for initial assumption and final choice)
    assert history.generations_count >= num_iterations // 3 * 1.2 + 2
    # metric improved
    assert np.mean([ind.fitness.value for ind in history.generations[0].data]) > \
           np.mean([ind.fitness.value for ind in history.generations[-1].data])
