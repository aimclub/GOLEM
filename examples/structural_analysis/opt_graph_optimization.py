import os.path
import random

from typing import Callable

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
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


def custom_metric(graph: OptGraph, visualisation: bool = False) -> float:
    """ Get toy metric for demonstration. """
    if visualisation:
        graph.show()
    metric = -1*random.randint(80, 100)/100
    return metric


if __name__ == "__main__":
    opt_graph = get_opt_graph()
    opt_graph.show()

    objective = Objective({'custom': custom_metric})
    node_factory = DefaultOptNodeFactory()

    opt_graph = get_opt_graph()
    requirements = StructuralAnalysisRequirements(graph_verifier=GraphVerifier(DEFAULT_DAG_RULES),
                                                  seed=1)

    # structural analysis will optimize given graph if at least one of the metrics was increased.
    sa = GraphStructuralAnalysis(objectives=[objective] * 2, node_factory=node_factory,
                                 requirements=requirements)

    optimized_graph = sa.optimize(graph=opt_graph, n_jobs=1, max_iter=5)
    optimized_graph.show()
