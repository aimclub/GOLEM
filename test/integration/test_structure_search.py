from datetime import timedelta
from math import ceil

import pytest

from examples.synthetic_graph_evolution import experiment
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.metrics.edit_distance import tree_edit_dist


@pytest.mark.parametrize('size', [8,  16])
def test_simple_targets_are_found(size):
    target_graph = generate_labeled_graph('tree', size, node_labels=['x'])
    # running the example
    found_graph, history = experiment.run_trial(target_graph=target_graph,
                                                optimizer_setup=tree_search_setup,
                                                timeout=timedelta(minutes=1))

    assert found_graph is not None

    found_nx_graph = BaseNetworkxAdapter().restore(found_graph)
    allowed_error = ceil(size * 0.2)  # 20% of target size
    distance = tree_edit_dist(target_graph, found_nx_graph)

    assert distance <= allowed_error
