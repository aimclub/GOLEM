from copy import deepcopy
from typing import Optional, Dict, Any

import networkx as nx

from examples.molecule_search.mol_graph import MolGraph
from examples.synthetic_graph_evolution.generators import nx_to_directed
from golem.core.adapter import BaseOptimizationAdapter
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.graph import OptGraph


class MolAdapter(BaseOptimizationAdapter):
    def __init__(self):
        super().__init__(base_graph_class=MolGraph)
        self.nx_adapter = BaseNetworkxAdapter()

    def _restore(self, opt_graph: OptGraph, metadata: Optional[Dict[str, Any]] = None) -> MolGraph:
        digraph = self.nx_adapter.restore(opt_graph)
        # to ensure backward compatibility with old individuals without 'nxid' field in nodes
        if not any(x is None for x in list(dict(digraph.nodes(data='nxid')).values())):
            # return to previous node indexing
            digraph = nx.relabel_nodes(digraph, dict(digraph.nodes(data='nxid')))
        digraph = restore_edges_params_from_nodes(digraph)
        nx_graph = digraph.to_undirected()
        mol_graph = MolGraph.from_nx_graph(nx_graph)
        return mol_graph

    def _adapt(self, adaptee: MolGraph) -> OptGraph:
        nx_graph = adaptee.get_nx_graph()
        digraph = nx_to_directed(nx_graph)
        digraph = store_edges_params_in_nodes(digraph)
        opt_graph = self.nx_adapter.adapt(digraph)
        return opt_graph


def store_edges_params_in_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    graph = deepcopy(graph)
    edges_params = {}
    for node in graph.nodes():
        edge_params = {}
        for predecessor in graph.predecessors(node):
            edge_params.update({str(predecessor): graph.get_edge_data(predecessor, node)})
        edges_params.update({node: edge_params})
    nx.set_node_attributes(graph, edges_params, name='edges_params')
    return graph


def restore_edges_params_from_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    graph = deepcopy(graph)
    edge_params_by_node = nx.get_node_attributes(graph, 'edges_params')
    all_edges_params = {}
    for node in graph.nodes():
        for predecessor in graph.predecessors(node):
            node_params = edge_params_by_node[node]
            # sometimes by unknown reason some nodes are encoded as int, some as str.
            # maybe that's deserialization messing up somewhere.
            edge_params = node_params.get(predecessor) or node_params.get(str(predecessor))
            if edge_params:
                all_edges_params[(predecessor, node)] = edge_params
    nx.set_edge_attributes(graph, all_edges_params)
    return graph
