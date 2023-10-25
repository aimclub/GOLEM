from copy import deepcopy

import pytest
from hyperopt import hp

from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.search_space import SearchSpace
from golem.core.tuning.sequential import SequentialTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from test.unit.mocks.common_mocks import MockAdapter, MockDomainStructure, MockNode, MockObjectiveEvaluate, \
    mock_graph_with_params, opt_graph_with_params
from test.unit.optimizers.test_composing_history import check_individuals_in_history
from test.unit.utils import ParamsProductMetric, ParamsSumMetric


def not_tunable_mock_graph():
    node_d = MockNode('d')
    node_final = MockNode('f', nodes_from=[node_d])
    graph = MockDomainStructure([node_final])

    return graph


@pytest.fixture()
def search_space():
    params_per_operation = {
        'a': {
            'a1': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 7],
                'type': 'discrete'
            },
            'a2': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'
            }
        },
        'b': {
            'b1': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["first", "second", "third"]],
                'type': 'categorical'
            },
            'b2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
        },
        'e': {
            'e1': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
            'e2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            }
        },
        'k': {
            'k': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-2, 10.0],
                'type': 'continuous'
            }
        }}
    return SearchSpace(params_per_operation)


@pytest.mark.parametrize('tuner_cls', [OptunaTuner, SimultaneousTuner, SequentialTuner, IOptTuner])
@pytest.mark.parametrize('graph, adapter, obj_eval',
                         [(mock_graph_with_params(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value}))),
                          (opt_graph_with_params(), None,
                           ObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value})))])
def test_tuner_improves_metric(search_space, tuner_cls, graph, adapter, obj_eval):
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=20)
    tuned_graph = tuner.tune(deepcopy(graph))
    assert tuned_graph is not None
    assert tuner.obtained_metric is not None
    assert tuner.init_metric > tuner.obtained_metric


@pytest.mark.parametrize('tuner_cls', [OptunaTuner, SimultaneousTuner, SequentialTuner, IOptTuner])
@pytest.mark.parametrize('graph, adapter, obj_eval',
                         [(not_tunable_mock_graph(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value})))])
def test_tuner_with_no_tunable_params(search_space, tuner_cls, graph, adapter, obj_eval):
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=20)
    tuned_graph = tuner.tune(deepcopy(graph))
    assert tuned_graph is not None
    assert tuner.obtained_metric is not None
    assert tuner.init_metric == tuner.obtained_metric


@pytest.mark.parametrize('graph', [mock_graph_with_params(), opt_graph_with_params(), not_tunable_mock_graph()])
def test_node_tuning(search_space, graph):
    obj_eval = MockObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value}))
    adapter = MockAdapter()
    for node_idx in range(graph.length):
        tuner = SequentialTuner(obj_eval, search_space, adapter, iterations=10)
        tuned_graph = tuner.tune_node(graph, node_idx)
        assert tuned_graph is not None
        assert tuner.obtained_metric is not None
        assert tuner.init_metric >= tuner.obtained_metric


@pytest.mark.parametrize('tuner_cls', [OptunaTuner])
@pytest.mark.parametrize('init_graph, adapter, obj_eval',
                         [(mock_graph_with_params(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value,
                                                            'prod_metric': ParamsProductMetric.get_value},
                                                           is_multi_objective=True))),
                          (opt_graph_with_params(), None,
                           ObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value,
                                                        'prod_metric': ParamsProductMetric.get_value},
                                                       is_multi_objective=True)))])
def test_multi_objective_tuning(search_space, tuner_cls, init_graph, adapter, obj_eval):
    init_metric = obj_eval.evaluate(init_graph)
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=20, objectives_number=2)
    tuned_graphs = tuner.tune(deepcopy(init_graph), show_progress=False)
    for graph in tuned_graphs:
        assert type(graph) == type(init_graph)
        final_metric = obj_eval.evaluate(graph)
        assert final_metric is not None
        assert not init_metric.dominates(final_metric)


@pytest.mark.parametrize('tuner_cls', [OptunaTuner, SimultaneousTuner, SequentialTuner, IOptTuner])
@pytest.mark.parametrize('graph, adapter, obj_eval',
                         [(mock_graph_with_params(), MockAdapter(),
                           MockObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value}))),
                          (opt_graph_with_params(), None,
                           ObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value})))])
def test_tuning_supports_history(search_space, tuner_cls, graph, adapter, obj_eval):
    history = OptHistory()
    iterations = 10
    tuner = tuner_cls(obj_eval, search_space, adapter, iterations=iterations, history=history)
    tuner.tune(deepcopy(graph))
    assert history.tuning_result is not None
    check_individuals_in_history(history)
