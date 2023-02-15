from datetime import timedelta
from typing import Optional, Callable, Dict

import networkx as nx
import numpy as np
from networkx import graph_edit_distance

from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.metrics.graph_metrics import min_max
from libs.netcomp import edit_distance


def get_edit_dist_metric(target_graph: nx.DiGraph,
                         timeout=timedelta(seconds=60),
                         upper_bound: Optional[int] = None,
                         requirements: Optional[GraphRequirements] = None,
                         ) -> Callable[[nx.DiGraph], float]:
    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        operations_do_match = node_content_1.get('name') == node_content_2.get('name')
        return True or operations_do_match

    if requirements:
        upper_bound = upper_bound or int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = timeout or requirements.max_graph_fit_time

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.total_seconds() if timeout else None,
                                 )
        return ged or upper_bound

    return metric


def matrix_edit_dist(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)
    nmin, nmax = min_max(target_adj.shape[0], adj.shape[0])
    if nmin != nmax:
        shape = (nmax, nmax)
        target_adj.resize(shape)
        adj.resize(shape)
    value = edit_distance(target_adj, adj)
    return value
