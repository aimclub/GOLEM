import os.path
from itertools import chain

import pytest

from golem.core.dag.graph import Graph
from golem.core.dag.graph_node import GraphNode
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.paths import project_root


@pytest.mark.parametrize('history_path', [
    'test/data/history_composite_bn_healthcare.json',
])
def test_external_history_load(history_path):
    """The idea is that external histories must be loadable by GOLEM.
    External histories are those by projects that depend on GOLEM, e.g. BAMT.

    This is needed so that GOLEM could be used as a stand-alone
    analytic tool for external histories. Or, or example, external histories
    could be used in FEDOT.Web that depends only on GOLEM.
    """
    history_path = project_root() / history_path

    assert os.path.exists(history_path)

    # Expect load without errors
    history: OptHistory = OptHistory.load(history_path)

    assert history is not None
    history_plausible(history)
    assert len(history.generations) > 0


def history_plausible(history: OptHistory):
    def individual_plausible(individual: Individual):
        graph_correct = isinstance(individual.graph, (Graph, dict))
        nodes_are_correct = True
        if not isinstance(individual.graph, dict):
            nodes_are_correct = all(isinstance(node, (GraphNode, dict)) for node in individual.graph.nodes)
        fitness_correct = isinstance(individual.fitness, Fitness)
        parent_operator = individual.parent_operator
        operations_correct = True
        if parent_operator:
            type_correct = parent_operator.type_ in ['mutation', 'crossover', 'selection', 'tuning']
            parent_inds_correct = all(isinstance(ind, Individual) for ind in parent_operator.parent_individuals)
            operations_correct = type_correct and parent_inds_correct
        assert graph_correct
        assert nodes_are_correct
        assert fitness_correct
        assert operations_correct

    all_historical_fitness_correct = all(isinstance(value, float) for value in history.all_historical_fitness)
    assert all_historical_fitness_correct
    for individual in chain(*history.generations):
        individual_plausible(individual)
