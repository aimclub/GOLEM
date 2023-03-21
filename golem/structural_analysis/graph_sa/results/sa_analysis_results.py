from typing import List, Tuple, Callable, Optional

from golem.core.dag.graph import Graph, GraphNode
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.results.object_sa_result import ObjectSAResult


class SAAnalysisResults:
    """ Class presenting results of Structural Analysis for the whole graph. """
    def __init__(self, graph: Graph,
                 nodes_to_analyze: List[GraphNode] = None, edges_to_analyze: List[Edge] = None):

        self.nodes_to_analyze = nodes_to_analyze or graph.nodes
        self.edges_to_analyze = edges_to_analyze or graph.get_edges()
        self.results = {'nodes': [], 'edges': []}

    @property
    def is_empty(self):
        """ Bool value indicating is there any calculated results. """
        if self.results['nodes'] is None and self.results['edges'] is None:
            return True
        return False

    def get_info_about_worst_result(self, metric_idx_to_optimize_by: int):
        """ Returns info about the worst result. """
        worst_value = None
        worst_result = None
        for i, res in enumerate(self.results['nodes'] + self.results['edges']):
            cur_res = res.get_worst_result_with_names(
                metric_idx_to_optimize_by=metric_idx_to_optimize_by)
            if not worst_value or cur_res['value'] > worst_value:
                worst_value = cur_res['value']
                worst_result = cur_res
        return worst_result

    def add_node_result(self, node_result: ObjectSAResult):
        """ Add calculated result for node. """
        self.results['nodes'].append(node_result)

    def add_edge_result(self, edge_result: ObjectSAResult):
        """ Add calculated result for edge. """
        self.results['nodes'].append(edge_result)
