import json
import os.path
from datetime import datetime
from typing import List, Tuple, Callable, Optional, Union, Dict, Any

from golem.core.dag.graph import Graph, GraphNode
from golem.core.log import default_log
from golem.core.paths import project_root
from golem.core.utilities.serializable import Serializable
from golem.serializers import Serializer
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.results.object_sa_result import ObjectSAResult, \
    StructuralAnalysisResultsRepository
from golem.structural_analysis.graph_sa.results.utils import get_entity_by_str


class SAAnalysisResults:
    """ Class presenting results of Structural Analysis for the whole graph. """

    def __init__(self, graph: Graph,
                 nodes_to_analyze: List[GraphNode] = None, edges_to_analyze: List[Edge] = None):

        self.nodes_to_analyze = nodes_to_analyze or graph.nodes
        self.edges_to_analyze = edges_to_analyze or graph.get_edges()
        self.results = {'nodes': [], 'edges': []}
        self.log = default_log('sa_results')

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

    def add_result(self, result: ObjectSAResult):
        if isinstance(result.entity, Edge):
            self.add_edge_result(edge_result=result)
        else:
            self.add_node_result(node_result=result)


    def add_node_result(self, node_result: ObjectSAResult):
        """ Add calculated result for node. """
        self.results['nodes'].append(node_result)

    def add_edge_result(self, edge_result: ObjectSAResult):
        """ Add calculated result for edge. """
        self.results['edges'].append(edge_result)

    def save(self, path: str = None, datetime_in_path: bool = True) -> dict:
        dict_results = dict()
        for entity_type in self.results.keys():
            if entity_type not in dict_results.keys():
                dict_results[entity_type] = {}
            for entity in self.results[entity_type]:
                dict_results[entity_type].update(entity.get_dict_results())

        json_data = json.dumps(dict_results, cls=Serializer)

        if not path:
            path = os.path.join(project_root(), 'sa_results.json')
        if datetime_in_path:
            file_name = os.path.basename(path).split('.')[0]
            file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{file_name}.json"
            path = os.path.join(os.path.dirname(path), file_name)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_data)
            self.log.debug(f'SA results saved in the path: {path}.')

        return dict_results

    @staticmethod
    def load(source: Union[str, dict], graph: Optional[Graph] = None):
        if isinstance(source, str):
            source = json.load(open(source))

        sa_result = SAAnalysisResults(graph=graph)
        results_repo = StructuralAnalysisResultsRepository()
        for key in source.keys():
            for entity_result in source[key]:
                entity = get_entity_by_str(graph=graph, entity_str=entity_result)
                cur_result = ObjectSAResult(entity=entity)
                for result in source[key][entity_result]:
                    result_class = results_repo.get_class_by_str(result_str=result)
                    result_approach = result_class()
                    result_value = source[key][entity_result][result]
                    if isinstance(result_value, dict):
                        # for replacement operations
                        entities = [get_entity_by_str(graph=graph, entity_str=key) for key in result_value.keys()]
                        for i, entity in enumerate(entities):
                            result_approach.add_results(entity, list(result_value)[i])
                    else:
                        # for deletion operations
                        result_approach.add_results(result_value)
                    cur_result.add_result(result_approach)
                sa_result.add_result(cur_result)
        return sa_result


