from typing import List, Tuple, Callable, Optional

from golem.core.optimisers.graph import OptGraph, OptNode
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.result_presenting_structures.object_sa_result import ObjectSAResult


class SAAnalysisResults:
    def __init__(self, graph: OptGraph,
                 nodes_to_analyze: List[OptNode] = None, edges_to_analyze: List[Edge] = None,
                 nodes_approaches: List[Callable] = None, edges_approaches: List[Callable] = None):

        self.nodes_to_analyze = nodes_to_analyze or graph.nodes
        self.edges_to_analyze = edges_to_analyze or graph.get_edges()
        self.results = {'nodes': [], 'edges': []}
        if nodes_approaches:
            self.results['nodes'] = [ObjectSAResult(approaches=[approach.__name__ for approach in nodes_approaches])
                                     for _ in nodes_to_analyze]
        if edges_approaches:
            self.results['edges'] = [ObjectSAResult(approaches=[approach.__name__ for approach in edges_approaches])
                                     for _ in edges_to_analyze]

    @property
    def is_empty(self):
        """ Bool value indicating is there any calculated results. """
        if self.results['nodes'] is None and self.results['edges'] is None:
            return True
        return False

    def get_worst_result(self) -> float:
        """ Worst result among all nodes and all approaches. """
        return max([res.get_worst_result() for res in self.results['nodes']+self.results['edges']])

    def get_worst_result_with_names(self) -> dict:
        """ Returns worst result with additional information. """
        worst_value = self.get_worst_result()
        for res in self.results['nodes'] + self.results['edges']:
            if res.get_worst_result() == worst_value:
                return res.get_worst_result_with_names()

    def get_info_about_worst_result(self):
        worst_value = self.get_worst_result()
        for i, res in enumerate(self.results['nodes'] + self.results['edges']):
            if res.get_worst_result() == worst_value:
                result = {'entity': res.entity}
                result.update(res.get_worst_result_with_names())
                return result

    def add_nodes_result(self, nodes_result):
        self.results['nodes'] = nodes_result

    def add_edges_result(self, edges_result):
        self.results['edges'] = edges_result

    def add_node_result(self, node_result):
        self.results['nodes'].append(node_result)

    def add_edge_result(self, edge_result):
        self.results['nodes'].append(edge_result)
