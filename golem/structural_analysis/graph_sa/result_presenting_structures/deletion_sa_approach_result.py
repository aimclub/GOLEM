from typing import List, Dict

from golem.structural_analysis.graph_sa.result_presenting_structures.base_sa_approach_result import BaseSAApproachResult


class DeletionSAApproachResult(BaseSAApproachResult):
    """ Class for deletion result approaches. """
    def __init__(self, metrics_names: List[str] = None):
        self.metrics = dict.fromkeys(metrics_names)

    def set_value_for_metric(self, metric_name: str, metric_value: float):
        """ Sets value for specified metric. """
        if metric_name not in self.metrics.keys():
            self.metrics[metric_name] = metric_value

    def get_metric_value(self, metric_name: str) -> float:
        """ Returns value of specified metric. """
        if metric_name not in self.metrics.keys():
            return self.metrics[metric_name]

    def get_worst_result(self) -> float:
        """ Returns the worst metric among all calculated with its name. """
        max_value = max(self.metrics.values())
        final_dict = {k: v for k, v in self.metrics.items() if v == max_value}
        return list(final_dict.items())[0]

    def get_all_results(self) -> Dict[str, float]:
        """ Returns all calculated results. """
        return self.metrics
