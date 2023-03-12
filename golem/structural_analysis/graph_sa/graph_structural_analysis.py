from copy import deepcopy
from typing import List, Callable, Any, Optional
import multiprocessing

from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.structural_analysis.graph_sa.edge_sa_approaches import EdgeAnalyzeApproach
from golem.structural_analysis.graph_sa.edges_analysis import EdgesAnalysis
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeAnalyzeApproach
from golem.structural_analysis.graph_sa.nodes_analysis import NodesAnalysis
from golem.structural_analysis.graph_sa.sa_approaches_repository import StructuralAnalysisApproachesRepository
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements


class GraphStructuralAnalysis:
    """
    This class works as facade and allows to apply all kind of approaches
    to whole graph and separate nodes together.

    :param objectives: list of objective functions for computing metric values
    :param node_factory: node factory to advise changes from available operations and models
    :param approaches: methods applied to graph. Default: None
    :param requirements: extra requirements to define specific details for different approaches.\
    See StructuralAnalysisRequirements class documentation.
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural/
    Default: False
    """

    def __init__(self, objectives: List[Callable],
                 node_factory: OptNodeFactory,
                 is_preproc: bool = True,
                 approaches: List = None,
                 requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):

        self.is_preproc = is_preproc

        self._log = default_log(self)

        if approaches:
            self.nodes_analyze_approaches = [approach for approach in approaches
                                             if issubclass(approach, NodeAnalyzeApproach)]
            self.edges_analyze_approaches = [approach for approach in approaches
                                             if issubclass(approach, EdgeAnalyzeApproach)]
        else:
            self._log.message('Approaches for analysis are not given, thus will be set to defaults.')
            self.nodes_analyze_approaches = None
            self.edges_analyze_approaches = None

        self._nodes_analyze = NodesAnalysis(objectives=objectives,
                                            node_factory=node_factory,
                                            approaches=self.nodes_analyze_approaches,
                                            requirements=requirements,
                                            path_to_save=path_to_save)

        self._edges_analyze = EdgesAnalysis(objectives=objectives,
                                            approaches=self.edges_analyze_approaches,
                                            requirements=requirements,
                                            path_to_save=path_to_save)
        
        self._log = default_log('SA')

    def analyze(self, graph: OptGraph,
                nodes_to_analyze: List[OptNode] = None, edges_to_analyze: List[Edge] = None,
                n_jobs: int = -1, timer: OptimisationTimer = None):
        """
        Applies defined structural analysis approaches

        :param graph: graph object to analyze
        :param nodes_to_analyze: nodes to analyze. Default: all nodes
        :param edges_to_analyze: edges to analyze. Default: all edges
        :param n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
        :param timer: timer with timeout left for optimization
        """

        result = dict()

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if self.is_preproc:
            graph = self.graph_preprocessing(graph=graph)

        if self.nodes_analyze_approaches:
            result['nodes_result'] = self._nodes_analyze.analyze(graph=graph,
                                                                 nodes_to_analyze=nodes_to_analyze,
                                                                 n_jobs=n_jobs, timer=timer)

        if self.edges_analyze_approaches:
            result['edges_result'] = self._edges_analyze.analyze(graph=graph,
                                                                 edges_to_analyze=edges_to_analyze,
                                                                 n_jobs=n_jobs, timer=timer)

        return result

    def optimize(self, n_jobs: int = -1, timer: OptimisationTimer = None, save_path: str = None) -> OptGraph:
        """ Optimizes graph by applying 'analyze' method and deleting/replacing parts
        of graph iteratively """
        pass

    @staticmethod
    def apply_results(graph: OptGraph, analysis_result: Optional[dict] = None) -> OptGraph:
        """ Optimizes graph by applying actions specified in analysis_result """
        pass

    @staticmethod
    def graph_preprocessing(graph: OptGraph):
        """ Graph preprocessing, which consists in removing consecutive nodes
        with the same models/operations in the graph """
        for node_child in reversed(graph.nodes):
            if not node_child.nodes_from or len(node_child.nodes_from) != 1:
                continue
            nodes_to_delete = []
            for node_parent in node_child.nodes_from:
                if node_child.operation.operation_type == node_parent.operation.operation_type:
                    nodes_to_delete.append(node_parent)
            for node in nodes_to_delete:
                graph.delete_node(node)
        return graph
