from typing import List, Callable, Tuple, Optional, Union

from golem.core.dag.graph_node import GraphNode
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.result_presenting_structures.base_sa_approach_result import BaseSAApproachResult
from golem.structural_analysis.graph_sa.result_presenting_structures.deletion_sa_approach_result import \
    DeletionSAApproachResult
from golem.structural_analysis.graph_sa.result_presenting_structures.replace_sa_approach_result import \
    ReplaceSAApproachResult

NODE_DELETION = 'NodeDeletionAnalyze'
NODE_REPLACEMENT = 'NodeReplaceOperationAnalyze'
SUBTREE_DELETION = 'SubtreeDeletionAnalyze'
EDGE_DELETION = 'EdgeDeletionAnalyze'
EDGE_REPLACEMENT = 'EdgeReplaceOperationAnalyze'


class StructuralAnalysisApproachesRepository:
    approaches_dict = {NODE_DELETION: {'result_class': DeletionSAApproachResult},
                       NODE_REPLACEMENT: {'result_class': ReplaceSAApproachResult},
                       SUBTREE_DELETION: {'result_class': DeletionSAApproachResult},
                       EDGE_DELETION: {'result_class': DeletionSAApproachResult},
                       EDGE_REPLACEMENT: {'result_class': ReplaceSAApproachResult}}

    def get_method_by_result_class(self, result_class: BaseSAApproachResult, entity_class: str) -> str:
        for method in self.approaches_dict.keys():
            if self.approaches_dict[method]['result_class'] == result_class.__class__ \
                    and entity_class in method.lower():
                return method


class ObjectSAResult:
    """ Class specifying results of Structural Analysis for one entity(node or edge). """
    def __init__(self, entity: Union[GraphNode, Edge]):
        self.entity = entity
        self.result_approaches: List[BaseSAApproachResult] = []

    def get_worst_result(self, metric_idx_to_optimize_by: int) -> float:
        """ Returns the worst result among all result classes. """
        worst_results = []
        for approach in self.result_approaches:
            worst_results.append(approach.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by))
        if not worst_results:
            return 0
        return max(worst_results)

    def get_worst_result_with_names(self, metric_idx_to_optimize_by: int) -> dict:
        """ Returns worst result with additional information. """
        worst_result = self.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by)
        for app in self.result_approaches:
            if app.get_worst_result(metric_idx_to_optimize_by=metric_idx_to_optimize_by) == worst_result:
                entity_type = 'edge' if isinstance(self.entity, Edge) else 'node'
                sa_approach_name = StructuralAnalysisApproachesRepository()\
                    .get_method_by_result_class(app, entity_type)
                result = {'entity': self.entity, 'approach_name': sa_approach_name}
                result.update(app.get_worst_result_with_names(metric_idx_to_optimize_by=metric_idx_to_optimize_by))
                return result

    def add_result(self, result: BaseSAApproachResult):
        self.result_approaches.append(result)
