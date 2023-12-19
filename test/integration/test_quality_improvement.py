from datetime import timedelta
from typing import Sequence

import numpy as np
import pytest

from examples.synthetic_graph_evolution.experiment_setup import run_trial
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from examples.synthetic_graph_evolution.tree_search import tree_search_setup
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.utilities.utils import set_random_seed


def run_graph_trial(optimizer_cls):
    # input data initialization
    target_graph = generate_labeled_graph('gnp', 20, node_labels=['X', 'Y'])
    # running the example
    return run_trial(target_graph=target_graph,
                     optimizer_setup=graph_search_setup,
                     optimizer_cls=optimizer_cls,
                     timeout=timedelta(minutes=1))


def run_tree_trial(optimizer_cls):
    # input data initialization
    target_graph = generate_labeled_graph('tree', 10, node_labels=['X', 'Y'])
    # running the example
    return run_trial(target_graph=target_graph,
                     optimizer_setup=tree_search_setup,
                     optimizer_cls=optimizer_cls,
                     timeout=timedelta(minutes=1))


@pytest.mark.parametrize('run_fun', [run_graph_trial, run_tree_trial])
@pytest.mark.parametrize('optimizer_cls', [EvoGraphOptimizer])
def test_multiobjective_improvement(optimizer_cls, run_fun):
    set_random_seed(42)
    found_graph, history = run_fun(optimizer_cls)
    quality_improved, complexity_improved = check_improvement(history)

    assert found_graph is not None
    assert found_graph.length > 1
    assert quality_improved
    assert complexity_improved


def check_improvement(history: OptHistory):
    first_pop = history.generations[1]
    pareto_front = history.archive_history[-1]
    first_pop_metrics = get_mean_metrics(first_pop)
    pareto_front_metrics = get_mean_metrics(pareto_front)

    quality_improved = pareto_front_metrics[0] < first_pop_metrics[0]
    complexity_improved = pareto_front_metrics[-1] < first_pop_metrics[-1]
    return quality_improved, complexity_improved


def get_mean_metrics(population) -> Sequence[float]:
    return np.mean([ind.fitness.values for ind in population], axis=0)
