from typing import List

from golem.structural_analysis.graph_sa.sa_approaches_repository import NODE_DELETION, NODE_REPLACEMENT, \
    SUBTREE_DELETION, EDGE_DELETION, EDGE_REPLACEMENT


class StructuralAnalysisApproachesRepository:
    approaches_dict = {NODE_DELETION: {'result_class': NodeDeletionAnalyze},
                       NODE_REPLACEMENT: {'result_class': NodeReplaceOperationAnalyze},
                       SUBTREE_DELETION: {'result_class': SubtreeDeletionAnalyze},
                       EDGE_DELETION: {'result_class': EdgeDeletionAnalyze},
                       EDGE_REPLACEMENT: {'result_class': EdgeReplaceOperationAnalyze}}


class NodeSAResult:
    def __init__(self, approaches: List[str] = None):
        self.approaches = approaches

    def get_worst_result(self):
        pass