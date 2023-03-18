from typing import List, Dict, Tuple, Any, Union

from golem.core.optimisers.graph import OptNode
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.result_presenting_structures.base_sa_approach_result import BaseSAApproachResult


class ReplaceSAApproachResult(BaseSAApproachResult):
    """ Class for presenting replacing result approaches. """
    def __init__(self):
        """ Main dictionary `self.metrics` contains entities as key and
        list with metrics as values"""
        self.entities_metrics = dict()

    def add_results(self, entity_to_replace_to: Union[OptNode, Edge], metrics_values: List[float]):
        """ Sets value for specified metric. """
        self.entities_metrics[entity_to_replace_to] = metrics_values

    def get_worst_result(self, metric_idx_to_optimize_by: int) -> float:
        """ Returns value of the worst metric. """
        return max([metrics[metric_idx_to_optimize_by] for metrics in list(self.entities_metrics.values())])

    def get_worst_result_with_names(self, metric_idx_to_optimize_by: int) -> dict:
        """ Returns the worst metric among all calculated with its name and node's to replace to name. """
        worst_value = self.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by)
        for entity in self.entities_metrics:
            if list(self.entities_metrics[entity])[metric_idx_to_optimize_by] == worst_value:
                return {'value': worst_value, 'entity_to_replace_to': entity}

    def get_all_results(self) -> Dict[str, float]:
        """ Returns all calculated results. """
        return self.entities_metrics
