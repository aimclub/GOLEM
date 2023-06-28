from datetime import timedelta
from functools import partial

from examples.synthetic_graph_evolution.experiment_setup import run_trial
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import STRUCTURAL_DIVERSITY_FREQUENCY_CHECK
from golem.metrics.edit_distance import tree_edit_dist
from golem.metrics.graph_metrics import degree_distance

DIVERSITY_THRESHOLD = 0.6


def set_up_params(gen_structural_check: int):
    """ It is possible to run test with and without structural check.
    To run test without structural test set `gen_structural_check` to -1,
    otherwise it has to be set to positive integer value. """
    target_graph = generate_labeled_graph('tree', 4, node_labels=['x'])
    objective = Objective(
        quality_metrics={'edit_dist': partial(tree_edit_dist, target_graph)},
        complexity_metrics={'degree': partial(degree_distance, target_graph)},
        is_multi_objective=False,
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=objective.is_multi_objective,
        mutation_types=[
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_drop,
        ],
        crossover_types=[CrossoverTypesEnum.none],
        gen_structural_check=gen_structural_check
    )
    return gp_params


def test_structural_diversity():
    """ Checks population's structural diversity. Diversity should not be lower than DIVERSITY_THRESHOLD. """
    target_graph = generate_labeled_graph('tree', 4, node_labels=['x'])
    node_types = ['x', 'y', 'z', 'w', 'v', 'u']
    gen_structural_check = STRUCTURAL_DIVERSITY_FREQUENCY_CHECK

    gp_params = set_up_params(gen_structural_check=gen_structural_check)
    # running the example
    _, history = run_trial(target_graph=target_graph,
                           optimizer_setup=partial(tree_search_setup, algorithm_parameters=gp_params),
                           node_types=node_types,
                           timeout=timedelta(minutes=1))

    h = history.generations[:-1]
    ratio_unique = [len(set(ind.graph.descriptive_id for ind in pop)) / len(pop) for pop in h]
    for i in range(len(ratio_unique)):
        if i % gen_structural_check == 0:
            # structural check is applied every 5-th generation
            assert ratio_unique[i] == 1
        else:
            assert ratio_unique[i] > DIVERSITY_THRESHOLD
