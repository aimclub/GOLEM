import json
import os
from copy import deepcopy
from typing import List, Callable, Any, Optional
import multiprocessing

from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import project_root
from golem.structural_analysis.graph_labels_using_sa import draw_nx_dag
from golem.structural_analysis.graph_sa.edge_sa_approaches import EdgeAnalyzeApproach, EdgeDeletionAnalyze, \
    EdgeReplaceOperationAnalyze
from golem.structural_analysis.graph_sa.edges_analysis import EdgesAnalysis
from golem.structural_analysis.graph_sa.entities.edge import Edge
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeAnalyzeApproach, NodeDeletionAnalyze, \
    NodeReplaceOperationAnalyze, SubtreeDeletionAnalyze
from golem.structural_analysis.graph_sa.nodes_analysis import NodesAnalysis
from golem.structural_analysis.graph_sa.result_presenting_structures.sa_analysis_results import SAAnalysisResults
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
            self.nodes_analyze_approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze,
                                             SubtreeDeletionAnalyze]
            self.edges_analyze_approaches = [EdgeDeletionAnalyze, EdgeReplaceOperationAnalyze]

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
        self.path_to_save = path_to_save

    def analyze(self, graph: OptGraph,
                nodes_to_analyze: List[OptNode] = None, edges_to_analyze: List[Edge] = None,
                n_jobs: int = -1, timer: OptimisationTimer = None) -> SAAnalysisResults:
        """
        Applies defined structural analysis approaches

        :param graph: graph object to analyze
        :param nodes_to_analyze: nodes to analyze. Default: all nodes
        :param edges_to_analyze: edges to analyze. Default: all edges
        :param n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
        :param timer: timer with timeout left for optimization
        """

        result = SAAnalysisResults(graph=graph)

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if self.is_preproc:
            graph = self.graph_preprocessing(graph=graph)

        if self.nodes_analyze_approaches:
            self._nodes_analyze.analyze(graph=graph,
                                        results=result,
                                        nodes_to_analyze=nodes_to_analyze,
                                        n_jobs=n_jobs, timer=timer)

        if self.edges_analyze_approaches:
            self._edges_analyze.analyze(graph=graph,
                                        results=result,
                                        edges_to_analyze=edges_to_analyze,
                                        n_jobs=n_jobs, timer=timer)

        return result

    def optimize(self, graph: OptGraph,
                 n_jobs: int = -1, timer: OptimisationTimer = None,
                 max_iter: int = 10) -> OptGraph:
        """ Optimizes graph by applying 'analyze' method and deleting/replacing parts
        of graph iteratively """

        approaches_repo = StructuralAnalysisApproachesRepository()
        approaches = self._nodes_analyze.approaches + self._edges_analyze.approaches
        approaches_names = [approach.__name__ for approach in approaches]

        # what actions were applied on the graph and how many
        actions_applied = dict.fromkeys(approaches_names, 0)

        graph_before_sa = deepcopy(graph)
        analysis_results = self.analyze(graph=graph, timer=timer, n_jobs=n_jobs)
        converged = False
        iter = 0

        if analysis_results.is_empty:
            self._log.message(f'0 actions were taken during SA')
            return graph

        while not converged:
            iter += 1
            worst_result = analysis_results.get_info_about_worst_result()
            if worst_result['value'] > 1.0:
                # apply the worst approach
                postproc_method = approaches_repo.postproc_method_by_name(worst_result['approach_name'])
                graph = postproc_method(graph=graph, worst_result=worst_result)
                actions_applied[f'{worst_result["approach_name"]}'] += 1

                if timer is not None and timer.is_time_limit_reached():
                    break

                if max_iter and iter >= max_iter:
                    break

                analysis_results = self.analyze(graph=graph, n_jobs=n_jobs,
                                                timer=timer)
            else:
                converged = True

        if self.path_to_save:
            _save_iteration_results(graph_before_sa=graph_before_sa, save_path=self.path_to_save)

        self._log.message(f'{iter} iterations passed during SA')
        self._log.message(f'The following actions were applied during SA: {actions_applied}')

        if isinstance(graph, OptGraph):
            return graph
        else:
            return graph_before_sa

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
            nodes_uid_to_delete = []
            for node_parent in node_child.nodes_from:
                if node_child.name == node_parent.name:
                    nodes_uid_to_delete.append(node_parent.uid)
            # there is a need to store nodes using uid since after deleting one of the nodes in graph
            # other nodes will not remain the same (nodes_from may be changed)
            for uid in nodes_uid_to_delete:
                node_to_delete = [node for node in graph.nodes if node.uid == uid][0]
                graph.delete_node(node_to_delete)
        return graph


def _save_iteration_results(graph_before_sa: OptGraph, save_path: str = None):
    """ Save visualizations for SA per iteration """
    json_path = os.path.join(save_path, 'results_per_iteration.json')
    graph_save_path = os.path.join(save_path, 'result_graphs')
    graph_before_sa.save(graph_save_path)
    if not os.path.exists(graph_save_path):
        os.makedirs(graph_save_path)
    try:
        draw_nx_dag(graph=graph_before_sa, save_path=save_path, json_path=json_path)
    except Exception as ex:
        log = default_log('draw_viz')
        log.error(f'Visualisation failed: {ex}')


def _save_iteration_results_to_json(analysis_results: dict, save_path: str = None):
    """ Save SA actions scores in json file """
    if save_path:
        save_path = os.path.join(save_path, 'results_per_iteration.json')
    else:
        save_path = os.path.join(project_root(), 'examples', 'structural_analysis',
                                 'show_sa_on_graph', 'results_per_iteration.json')
    if not os.path.exists(save_path):
        json_data = [analysis_results]
        with open(save_path, 'w') as file:
            file.write(json.dumps(json_data, indent=2, ensure_ascii=False))
    else:
        data = json.load(open(save_path))
        data.append(analysis_results)
        with open(save_path, 'w', encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
