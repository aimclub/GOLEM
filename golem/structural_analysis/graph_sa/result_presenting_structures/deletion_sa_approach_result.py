from typing import List, Dict

from golem.structural_analysis.graph_sa.result_presenting_structures.base_sa_approach_result import BaseSAApproachResult


class DeletionSAApproachResult(BaseSAApproachResult):
    """ Class for deletion result approaches. """
    def __init__(self):
        self.metrics = []

    def add_result(self, metric_value: float):
        """ Sets value for specified metric. """
        self.metrics.append(metric_value)

    def add_results(self, metrics_values: List[float]):
        self.metrics = metrics_values

    def get_metric_value(self, metric_idx: int) -> float:
        """ Returns value of specified metric. """
        return self.metrics[metric_idx]

    def get_worst_result(self) -> float:
        """ Returns the worst metric among all calculated. """
        return max(self.metrics)

    def get_worst_result_with_names(self) -> dict:
        return {'value': self.get_worst_result()}

    def get_all_results(self) -> List[float]:
        """ Returns all calculated results. """
        return self.metrics
