from datetime import timedelta
from functools import partial
from math import ceil

import numpy as np
import pytest
from typing import Tuple, Callable, Sequence

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.objective import Objective
from golem.metrics.edit_distance import tree_edit_dist, graph_size


def run_search(size: int, distance_function: Callable, timeout_min: int = 1) -> Tuple[float, Graph]:
    # defining task
    node_types = ['a', 'b']
    target_graph = generate_labeled_graph('tree', size, node_labels=node_types)
    objective = Objective(partial(distance_function, target_graph))

    # running the example
    optimizer, objective = tree_search_setup(objective=objective,
                                             timeout=timedelta(minutes=timeout_min),
                                             node_types=node_types)
    found_graphs = optimizer.optimise(objective)
    found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs

    # compute final distance. it accepts nx graphs, so first adapt it to accept our graphs
    adapted_dist = BaseNetworkxAdapter().adapt_func(distance_function)
    distance = adapted_dist(target_graph, found_graph)

    return distance, found_graph


@pytest.mark.parametrize('target_sizes, distance_function, indulgence',
                         [([10, 24], tree_edit_dist, 0.5),
                          ([30], graph_size, 0.1)])
def test_simple_targets_are_found(target_sizes, distance_function, indulgence):
    """ Checks if simple targets can be found within specified time. """
    for target_size in target_sizes:
        num_trials = 3
        distances = []
        for i in range(num_trials):
            distance, target_graph = run_search(target_size, distance_function=distance_function, timeout_min=1)
            distances.append(distance)

            assert target_graph is not None
            assert distance < target_size

        allowed_error = ceil(target_size * indulgence)
        mean_dist = np.mean(distances)

        assert mean_dist <= allowed_error
