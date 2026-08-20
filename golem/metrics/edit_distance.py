from datetime import timedelta
from typing import Optional, Callable, Dict, Sequence

import networkx as nx
import numpy as np
import zss
from networkx import graph_edit_distance, is_tree

from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.metrics.graph_metrics import min_max
from libs.netcomp import edit_distance


def _label_dist(label1: str, label2: str) -> int:
    return int(label1 != label2)


def tree_edit_dist(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    """Compares nodes by their `name` (if present) or `uid` attribute.
    Nodes with the same name/id are considered the same."""
    if not (is_tree(target_graph) and is_tree(graph)):
        raise ValueError('Both target graphs must be trees')
    target_tree_root = _nx_to_zss_tree(target_graph)
    cmp_tree_root = _nx_to_zss_tree(graph)
    dist = zss.simple_distance(target_tree_root, cmp_tree_root, label_dist=_label_dist)
    return dist


def graph_size(target_graph: nx.DiGraph, graph: nx.DiGraph) -> int:
    return abs(target_graph.number_of_nodes() - graph.number_of_nodes())


def _nx_to_zss_tree(graph: nx.DiGraph) -> zss.Node:
    # Root is the node without successors
    root = _get_root_node(graph)
    # that's why we first reverse the tree to get proper DFS traverse
    tree = graph.reverse()
    # Add nodes with appropriate labels for comparison
    nodes_dict = {}
    for node_id, node_data in tree.nodes(data=True):
        label = node_data.get('name', node_id)
        nodes_dict[node_id] = zss.Node(label)
    # Add edges
    for edge in tree.edges():
        nodes_dict[edge[0]].addkid(nodes_dict[edge[1]])
    return nodes_dict[root]


def _get_root_node(nxgraph: nx.DiGraph) -> Sequence:
    source = [n for (n, d) in nxgraph.out_degree() if d == 0][0]
    return source


def get_edit_dist_metric(target_graph: nx.DiGraph,
                         timeout=timedelta(seconds=60),
                         upper_bound: Optional[int] = None,
                         requirements: Optional[GraphRequirements] = None,
                         ) -> Callable[[nx.DiGraph], float]:
    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        return node_content_1.get('name') == node_content_2.get('name')

    if requirements:
        upper_bound = upper_bound or int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = timeout or requirements.max_graph_fit_time

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.total_seconds() if timeout else None,
                                  )
        return float(ged) or upper_bound

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
