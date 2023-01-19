import numpy as np
import pytest
from hyperopt import hp

from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.tuning.search_space import SearchSpace
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultanious_tuning import SimultaniousTuner
from test.unit.mocks.common_mocks import MockAdapter, MockObjectiveEvaluate, mock_graph_with_params, \
    opt_graph_with_params, MockNode, MockDomainStructure
from test.unit.utils import CustomMetric


def not_tunable_mock_graph():
    node_d = MockNode('d')
    node_final = MockNode('f', nodes_from=[node_d])
    graph = MockDomainStructure([node_final])

    return graph

@pytest.fixture()
def search_space():
    params_per_operation = {
        'a': {
            'a1': (hp.uniformint, [2, 7]),
            'a2': (hp.loguniform, [np.log(1e-3), np.log(1)])
        },
        'b': {
            'b1': (hp.choice, [["first", "second", "third"]]),
            'b2': (hp.uniform, [0.05, 1.0]),
        },
        'e': {
            'e1': (hp.uniform, [0.05, 1.0]),
            'e2': (hp.uniform, [0.05, 1.0])
        },
        'k': {
            'k': (hp.uniform, [1e-2, 10.0])
        }}
    return SearchSpace(params_per_operation)


@pytest.mark.parametrize('tuner_cls', [SimultaniousTuner, SequentialTuner])
@pytest.mark.parametrize('graph, adapter, obj_eval',
                         [(mock_graph_with_params(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value}))),
                          (not_tunable_mock_graph(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value}))),
                          (opt_graph_with_params(), None,
                           ObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value})))])
def test_general_tuner(search_space, tuner_cls, graph, adapter, obj_eval):
    init_metric = obj_eval.evaluate(graph)
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=20)
    tuned_graph = tuner.tune(graph)
    final_metric = obj_eval.evaluate(tuned_graph)
    assert final_metric is not None
    assert init_metric <= final_metric


@pytest.mark.parametrize('graph', [mock_graph_with_params(), opt_graph_with_params(), not_tunable_mock_graph()])
def test_node_tuning(search_space, graph):
    obj_eval = MockObjectiveEvaluate(Objective({'random_metric': CustomMetric.get_value}))
    adapter = MockAdapter()
    init_metric = obj_eval.evaluate(graph)
    for node_idx in range(graph.length):
        tuner = SequentialTuner(obj_eval, search_space, adapter, iterations=10)
        tuned_graph = tuner.tune_node(graph, node_idx)
        final_metric = obj_eval.evaluate(tuned_graph)
        assert final_metric is not None
        assert init_metric <= final_metric

