import random

import pandas as pd

from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_node_factory import DefaultOptNodeFactory
from golem.structural_analysis.graph_sa.graph_structural_analysis import GraphStructuralAnalysis


def get_opt_graph() -> OptGraph:
    node4 = OptNode({'name': 'node4'})
    node5 = OptNode({'name': 'node5'})
    node6 = OptNode({'name': 'node6'})
    node2 = OptNode({'name': 'node2'}, nodes_from=[node4, node5])
    node3 = OptNode({'name': 'node3'}, nodes_from=[node6])
    node1 = OptNode({'name': 'node1'}, nodes_from=[node2, node3])
    return OptGraph(node1)


def custom_metric(graph: OptGraph, visualisation: bool = False):
    if visualisation:
        graph.show()
    metric = -1*random.randint(80, 105)/100
    return metric


if __name__ == "__main__":
    opt_graph = get_opt_graph()
    # opt_graph.show()

    objective = Objective({'custom': custom_metric})
    node_factory = DefaultOptNodeFactory()

    sa = GraphStructuralAnalysis(objectives=[objective], node_factory=node_factory)

    sa.analyze(graph=opt_graph, n_jobs=1)

    # optimized_graph = sa.optimize(graph=opt_graph)
