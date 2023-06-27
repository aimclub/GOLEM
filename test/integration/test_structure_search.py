from datetime import timedelta
from math import ceil

import numpy as np
import pytest
from typing import Tuple

from examples.synthetic_graph_evolution.experiment_setup import run_trial
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph import Graph
from golem.metrics.edit_distance import tree_edit_dist


def run_search(size, timeout_min=1) -> Tuple[float, Graph]:
    target_graph = generate_labeled_graph('tree', size, node_labels=['x'])
    # running the example
    found_graph, history = run_trial(target_graph=target_graph,
                                     optimizer_setup=tree_search_setup,
                                     timeout=timedelta(minutes=timeout_min))

    found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
    distance = tree_edit_dist(target_graph, found_nx_graph)

    return distance, found_graph


@pytest.mark.parametrize('size', [10, 24])
def test_simple_targets_are_found(size):
    num_trials = 5
    distances = []
    for i in range(num_trials):
        distance, target_graph = run_search(size, timeout_min=2)
        distances.append(distance)

        assert target_graph is not None
        assert distance < size

    allowed_error = ceil(size * 0.25)  # 20% of target size
    mean_dist = np.mean(distances)

    assert mean_dist <= allowed_error
