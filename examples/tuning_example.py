from datetime import timedelta

from golem.core.optimisers.graph import OptNode, OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate, Objective
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.search_space import SearchSpace
from test.unit.utils import ParamsSumMetric


def opt_graph_with_params():
    node_a = OptNode('a')
    node_b = OptNode({'name': 'b', 'params': {'b2': 0.7, 'b3': 2}})
    node_c = OptNode('c', nodes_from=[node_a])
    node_d = OptNode('d', nodes_from=[node_b])
    node_final = OptNode('e', nodes_from=[node_c, node_d])
    graph = OptGraph(node_final)
    return graph


def get_search_space():
    params_per_operation = {
        'a': {
            'a1': {
                'sampling-scope': [2, 7],
                'type': 'discrete'
            },
            'a2': {
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'
            },
            'a3': {
                'sampling-scope': [['A', 'B', 'C']],
                'type': 'categorical'
            }
        },
        'b': {
            'b1': {
                'sampling-scope': [["first", "second", "third"]],
                'type': 'categorical'
            },
            'b2': {
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
        },
        'e': {
            'e1': {
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
            'e2': {
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            }
        },
        'k': {
            'k': {
                'sampling-scope': [1e-2, 10.0],
                'type': 'continuous'
            }
        }}
    return SearchSpace(params_per_operation)


if __name__ == '__main__':
    search_space = get_search_space()
    graph = opt_graph_with_params()
    # ищем такие параметры, чтобы их сумма была максимальна
    obj_eval = ObjectiveEvaluate(Objective({'sum_metric': ParamsSumMetric.get_value}))

    tuner = IOptTuner(obj_eval, search_space, iterations=10, n_jobs=1)
    tuned_graph = tuner.tune(graph)
