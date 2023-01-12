from datetime import timedelta
from typing import Sequence

import networkx as nx
import numpy as np
import pytest

from examples.synthetic_graph_evolution import abstract_graph_search
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


@pytest.mark.parametrize('optimizer_cls', [EvoGraphOptimizer])
def test_multiobjective_improvement(optimizer_cls):
    # input data initialization
    target_graph = nx.gnp_random_graph(20, p=0.15)

    # running the example
    found_graph, history, _ = abstract_graph_search.run_trial(target_graph=target_graph,
                                                              optimizer_cls=optimizer_cls,
                                                              timeout=timedelta(minutes=1))

    quality_improved, complexity_improved = check_improvement(history)

    assert found_graph is not None
    assert found_graph.length > 1
    assert quality_improved
    assert complexity_improved


def check_improvement(history: OptHistory):
    first_pop = history.individuals[1]
    pareto_front = history.archive_history[-1]

    first_pop_metrics = get_mean_metrics(first_pop)
    pareto_front_metrics = get_mean_metrics(pareto_front)

    quality_improved = pareto_front_metrics[0] < first_pop_metrics[0]
    complexity_improved = pareto_front_metrics[-1] < first_pop_metrics[-1]
    return quality_improved, complexity_improved


def get_mean_metrics(population) -> Sequence[float]:
    return np.mean([ind.fitness.values for ind in population], axis=0)
