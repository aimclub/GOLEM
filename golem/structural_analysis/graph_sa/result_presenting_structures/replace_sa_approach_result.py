from typing import List, Dict, Tuple, Any, Union

from golem.core.optimisers.graph import OptNode
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.result_presenting_structures.base_sa_approach_result import BaseSAApproachResult


class ReplaceSAApproachResult(BaseSAApproachResult):
    """ Class for replacing result approaches. """
    def __init__(self):
        """ Main dictionary `self.metrics` contains metrics names as keys and
        dict with {nodes_names: metric_value} as values"""
        self.metrics = dict()

    def add_results(self, entity_to_replace_to: Union[OptNode, Edge], metrics_values: List[float]):
        """ Sets value for specified metric. """
        self.metrics[entity_to_replace_to] = metrics_values

    def get_worst_result(self) -> float:
        """ Returns value of the worst metric. """
        return max(max(metric) for metric in self.metrics.values())

    # TODO: fix multi-objective optimization is incorrect now (add idx of metric to look for)
    def get_worst_result_with_names(self) -> dict:
        """ Returns the worst metric among all calculated with its name and node's to replace to name. """
        worst_value = self.get_worst_result()
        for entity in self.metrics:
            if max(self.metrics[entity]) == worst_value:
                return {'value': worst_value, 'entity_to_replace_to': entity}

    def get_all_results(self) -> Dict[str, float]:
        """ Returns all calculated results. """
        return self.metrics
