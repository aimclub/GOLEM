import os.path

from examples.structural_analysis.opt_graph_optimization import get_opt_graph, quality_custom_metric_1
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis
from golem.structural_analysis.graph_sa.results.sa_analysis_results import SAAnalysisResults
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements


TEST_FILE_NAME = 'test_sa_results.json'


def test_save_sa_results():
    opt_graph = OptGraph(OptNode('node1'))

    objective = Objective(
        quality_metrics={
            'quality_custom_1': quality_custom_metric_1,
        }
    )

    node_factory = DefaultOptNodeFactory()

    requirements = StructuralAnalysisRequirements(graph_verifier=GraphVerifier(DEFAULT_DAG_RULES),
                                                  main_metric_idx=0,
                                                  seed=1)

    # structural analysis will optimize given graph if at least one of the metrics was increased.
    sa = GraphStructuralAnalysis(objective=objective, node_factory=node_factory,
                                 requirements=requirements)

    graph, results = sa.optimize(graph=opt_graph, n_jobs=1, max_iter=1)

    path_to_save = os.path.join(TEST_FILE_NAME)
    saved_result = results.save(path=path_to_save, datetime_in_path=False)

    assert TEST_FILE_NAME in os.listdir()
    assert saved_result is not None


def test_load_sa_results():
    """ Results can be loaded in two ways """
    graph = get_opt_graph()
    path = os.path.join(TEST_FILE_NAME)
    result_load = SAAnalysisResults.load(source=path, graph=graph)

    assert result_load is not None
    os.remove(TEST_FILE_NAME)
