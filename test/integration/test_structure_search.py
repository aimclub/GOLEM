from datetime import timedelta
from math import ceil

import numpy as np
import pytest
from typing import Tuple, Callable

from examples.synthetic_graph_evolution.experiment_setup import run_trial
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.metrics.edit_distance import tree_edit_dist, graph_size


def run_search(size: int, distance_function: Callable, timeout_min: int = 1) -> Tuple[float, Graph]:
    target_graph = generate_labeled_graph('tree', size, node_labels=['x'])
    # running the example
    found_graph, history = run_trial(target_graph=target_graph,
                                     optimizer_setup=tree_search_setup,
                                     timeout=timedelta(minutes=timeout_min))

    found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
    distance = distance_function(target_graph, found_nx_graph)

    return distance, found_graph


@pytest.mark.parametrize('target_sizes, distance_function, indulgence',
                         [([10, 24], tree_edit_dist, 0.5),
                          ([10, 24], graph_size, 0.2)])
def test_simple_targets_are_found(target_sizes, distance_function, indulgence):
    """ Checks if simple targets can be found within specified time. """
    for target_size in target_sizes:
        num_trials = 5
        distances = []
        for i in range(num_trials):
            distance, target_graph = run_search(target_size, distance_function=distance_function, timeout_min=2)
            distances.append(distance)

            assert target_graph is not None
            assert distance < target_size

        allowed_error = ceil(target_size * indulgence)
        mean_dist = np.mean(distances)

        assert mean_dist <= allowed_error
