from typing import List, Dict, Tuple

from golem.structural_analysis.graph_sa.result_presenting_structures.base_sa_approach_result import BaseSAApproachResult


class ReplaceSAApproachResult(BaseSAApproachResult):
    """ Class for replacing result approaches. """
    def __init__(self, metrics_names: List[str] = None):
        """ Main dictionary `self.metrics` contains metrics names as keys and
        dict with {nodes_names: metric_value} as values"""
        self.metrics = dict.fromkeys(metrics_names)

    def set_value_for_metric_and_node(self, metric_name: str, node_name: str, metric_value: float):
        """ Sets value for specified metric. """
        if metric_name not in self.metrics.keys():
            return
        if self.metrics[metric_name] is None:
            self.metrics[metric_name] = {}
        self.metrics[metric_name][node_name] = metric_value

    def get_metric_value_for_node(self, metric_name: str, node_name: str) -> float:
        """ Returns value of specified metric and node. """
        if metric_name not in self.metrics.keys() or node_name not in self.metrics[metric_name].keys():
            return self.metrics[metric_name]
        return self.metrics[metric_name][node_name]

    def get_worst_result(self) -> float:
        """ Returns value of the worst metric. """
        return self.get_worst_result_with_names()[0]

    def get_worst_result_with_names(self) -> Tuple[float, str, str]:
        """ Returns the worst metric among all calculated with its name and node's to replace to name. """
        max_val = None
        worst_metric = None
        obj_to_replace = None
        for metric in self.metrics:
            nodes_and_metric_values = self.metrics[metric]
            for obj in nodes_and_metric_values:
                if not max_val:
                    max_val = nodes_and_metric_values[obj]
                    worst_metric = metric
                    obj_to_replace = obj
                else:
                    if nodes_and_metric_values[obj] > max_val:
                        max_val = nodes_and_metric_values[obj]
                        worst_metric = metric
                        obj_to_replace = obj
        return max_val, worst_metric, obj_to_replace

    def get_all_results(self) -> Dict[str, float]:
        """ Returns all calculated results. """
        return self.metrics
