from typing import List, Callable, Any
import multiprocessing

from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.timer import OptimisationTimer
from golem.structural_analysis.pipeline_sa.edge_sa_approaches import EdgeAnalyzeApproach
from golem.structural_analysis.pipeline_sa.edges_analysis import EdgesAnalysis
from golem.structural_analysis.pipeline_sa.entities.edge import Edge
from golem.structural_analysis.pipeline_sa.node_sa_approaches import NodeAnalyzeApproach
from golem.structural_analysis.pipeline_sa.nodes_analysis import NodesAnalysis
from golem.structural_analysis.pipeline_sa.sa_requirements import StructuralAnalysisRequirements


class GraphStructuralAnalysis:
    """
    This class works as facade and allows to apply all kind of approaches
    to whole pipeline and separate nodes together.

    :param pipeline: pipeline object to analyze
    :param objectives: list of objective functions for computing metric values
    :param task_type: type of solving task
    :param approaches: methods applied to pipeline. Default: None
    :param nodes_to_analyze: nodes to analyze. Default: all nodes
    :param requirements: extra requirements to define specific details for different approaches.\
    See StructuralAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural/
    Default: False
    """

    def __init__(self, pipeline: OptGraph, objectives: List[Callable],
                 task_type: Any,
                 is_preproc: bool = True,
                 approaches: List = None,
                 nodes_to_analyze: List[OptNode] = None,
                 edges_to_analyze: List[Edge] = None,
                 requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):

        if is_preproc:
            self.pipeline = self.pipeline_preprocessing(pipeline=pipeline)
        else:
            self.pipeline = pipeline

        self.log = default_log(self)

        if approaches:
            self.nodes_analyze_approaches = [approach for approach in approaches
                                             if issubclass(approach, NodeAnalyzeApproach)]
            self.edges_analyze_approaches = [approach for approach in approaches
                                             if issubclass(approach, EdgeAnalyzeApproach)]
            self.pipeline_analyze_approaches = [approach for approach in approaches
                                                if not (issubclass(approach, NodeAnalyzeApproach) or
                                                        issubclass(approach, EdgeAnalyzeApproach))]
        else:
            self.log.message('Approaches for analysis are not given, thus will be set to defaults.')
            self.nodes_analyze_approaches = None
            self.edges_analyze_approaches = None
            self.pipeline_analyze_approaches = None

        self._nodes_analyze = NodesAnalysis(pipeline=self.pipeline, objectives=objectives,
                                            task_type=task_type,
                                            approaches=self.nodes_analyze_approaches, requirements=requirements,
                                            path_to_save=path_to_save, nodes_to_analyze=nodes_to_analyze)

        self._edges_analyze = EdgesAnalysis(pipeline=self.pipeline, objectives=objectives,
                                            approaches=self.edges_analyze_approaches,
                                            requirements=requirements,
                                            edges_to_analyze=edges_to_analyze,
                                            path_to_save=path_to_save)

    def analyze(self, n_jobs: int = -1, timer: OptimisationTimer = None):
        """
        Applies defined structural analysis approaches
        """

        result = dict()

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if self.nodes_analyze_approaches:
            result['nodes_result'] = self._nodes_analyze.analyze(n_jobs=n_jobs, timer=timer)

        if self.edges_analyze_approaches:
            result['edges_result'] = self._edges_analyze.analyze(n_jobs=n_jobs, timer=timer)

        return result

    def optimize(self):
        pass

    @staticmethod
    def pipeline_preprocessing(pipeline: OptGraph):
        """ Pipeline preprocessing, which consists in removing consecutive nodes
        with the same models/operations in the pipeline """
        for node_child in reversed(pipeline.nodes):
            if not node_child.nodes_from or len(node_child.nodes_from) != 1:
                continue
            nodes_to_delete = []
            for node_parent in node_child.nodes_from:
                if node_child.operation.operation_type == node_parent.operation.operation_type:
                    nodes_to_delete.append(node_parent)
            for node in nodes_to_delete:
                pipeline.delete_node(node)
        return pipeline
