import os
import random
from functools import partial

from typing import Callable

from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.core.paths import project_root
from golem.metrics.graph_metrics import size_diff
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements


def get_opt_graph() -> OptGraph:
    """ Get diverse OptGraph. """
    node4 = OptNode({'name': 'node4'})
    node5 = OptNode({'name': 'node5'})
    node6 = OptNode({'name': 'node6'})
    node2 = OptNode({'name': 'node2'}, nodes_from=[node4, node5])
    node3 = OptNode({'name': 'node3'}, nodes_from=[node6])
    node1 = OptNode({'name': 'node1'}, nodes_from=[node2, node3])
    return OptGraph(node1)


def quality_custom_metric_1(_: OptGraph) -> float:
    """ Get toy metric for demonstration. """
    metric = -1 * random.randint(80, 100) / 100
    return metric


def quality_custom_metric_2(_: OptGraph) -> float:
    """ Get one more toy metric for demonstration. """
    metric = -1 * random.randint(70, 110) / 100
    return metric


def complexity_metric(graph: OptGraph, adapter: BaseNetworkxAdapter, metric: Callable) -> float:
    """ Adapts specified graph and returns metric calculated. """
    adapted_graph = adapter.restore(graph)
    return metric(adapted_graph)


if __name__ == "__main__":
    opt_graph = get_opt_graph()
    opt_graph.show()

    adapter = BaseNetworkxAdapter()
    objective = Objective(
        quality_metrics={
            'quality_custom_1': quality_custom_metric_1,
        },
        complexity_metrics={
            'graph_size': partial(complexity_metric,
                                  adapter=adapter, metric=partial(size_diff, adapter.restore(opt_graph))),
        },
        is_multi_objective=True
    )

    node_factory = DefaultOptNodeFactory()

    requirements = StructuralAnalysisRequirements(graph_verifier=GraphVerifier(DEFAULT_DAG_RULES),
                                                  main_metric_idx=0,
                                                  seed=1, replacement_number_of_random_operations_nodes=2,
                                                  replacement_number_of_random_operations_edges=2)

    path_to_save = os.path.join(project_root(), 'examples', 'sa')
    os.makedirs(path_to_save, exist_ok=True)
    # structural analysis will optimize given graph if at least one of the metrics was increased.
    sa = GraphStructuralAnalysis(objective=objective, node_factory=node_factory,
                                 requirements=requirements,
                                 path_to_save=path_to_save,
                                 is_visualize_per_iteration=False)

    graph, results = sa.optimize(graph=opt_graph, n_jobs=1, max_iter=3)

    # to show SA results on each iteration
    optimized_graph = GraphStructuralAnalysis.visualize_on_graph(graph=get_opt_graph(), analysis_result=results,
                                                                 metric_idx_to_optimize_by=0,
                                                                 mode="by_iteration",
                                                                 font_size_scale=0.6,
                                                                 save_path=path_to_save)

    graph.show()
