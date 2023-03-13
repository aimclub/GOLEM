import json
import os
from os.path import join
from typing import Optional, List, Type, Callable, Sequence, Any
import multiprocessing

from matplotlib import pyplot as plt

from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.graph_sa.node_sa_approaches import NodeAnalyzeApproach, NodeAnalysis
from golem.structural_analysis.graph_sa.postproc_methods import extract_result_values
from golem.structural_analysis.graph_sa.result_presenting_structures.object_sa_result import ObjectSAResult
from golem.structural_analysis.graph_sa.result_presenting_structures.sa_analysis_results import SAAnalysisResults
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements


class NodesAnalysis:
    """
    This class is for nodes structural analysis within a OptGraph .
    It takes nodes and approaches to be applied to chosen nodes.
    To define which nodes to analyze pass them to nodes_to_analyze filed
    or all nodes will be analyzed.

    :param objectives: objective functions for computing metric values
    :param node_factory: node factory to advise changes from available operations and models
    :param approaches: methods applied to nodes to modify the graph or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeReplaceOperationAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, objectives: List[Callable],
                 node_factory: OptNodeFactory,
                 approaches: Optional[List[Type[NodeAnalyzeApproach]]] = None,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):

        self.objectives = objectives
        self.node_factory = node_factory
        self.approaches = approaches
        self.requirements = \
            StructuralAnalysisRequirements() if requirements is None else requirements
        self.log = default_log(self)
        self.path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save

    def analyze(self, graph: OptGraph, results: SAAnalysisResults,
                nodes_to_analyze: List[OptNode] = None,
                n_jobs: int = -1, timer: OptimisationTimer = None) -> SAAnalysisResults:
        """
        Main method to run the analyze process for every node.

        :param graph: graph object to analyze
        :param results: SA results
        :param nodes_to_analyze: nodes to analyze. Default: all nodes
        :param n_jobs: n_jobs
        :param timer: timer indicating how much time is left for optimization
        :return nodes_results: dict with analysis result per OptNode
        """

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if not nodes_to_analyze:
            self.log.message('Nodes to analyze are not defined. All nodes will be analyzed.')
            nodes_to_analyze = graph.nodes

        # operation_types = []
        node_analysis = NodeAnalysis(approaches=self.approaches,
                                     approaches_requirements=self.requirements,
                                     node_factory=self.node_factory,
                                     path_to_save=self.path_to_save)

        with multiprocessing.Pool(processes=n_jobs) as pool:
            cur_results = pool.starmap(node_analysis.analyze,
                                       [[graph, node, self.objectives, timer]
                                        for node in nodes_to_analyze])[0]
            results.add_node_result(cur_results)

        # for node in nodes_to_analyze:
        #     results.add_node_result(node_analysis.analyze(graph, node, self.objectives, timer))
        #
        # for i, node in enumerate(nodes_to_analyze):
        #     nodes_results[f'id = {graph.nodes.index(node)}, '
        #                   f'operation = {node.name}'] = results[i]
        #     operation_types.append(f'{graph.nodes.index(node)}_{node.name}')
        #
        # if self.requirements.is_visualize:
        #     self._visualize_result_per_approach(nodes_results, operation_types)
        #
        #     if len(nodes_to_analyze) == len(graph.nodes):
        #         self._visualize_degree_correlation(graph, nodes_results)
        #
        # if self.requirements.is_save:
        #     self._save_results_to_json(nodes_results)
        return results

    def _save_results_to_json(self, result: dict):
        file_path = path_to_save_per_iter(root_path_to_save=self.path_to_save,
                                          file_name='nodes_SA_results',
                                          extension='json',
                                          folder_name='json_results')

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))

        self.log.message(f'Nodes Structural Analysis results were saved to {file_path}')

    def _visualize_result_per_approach(self, results: dict, types: list):
        gathered_results = extract_result_values(approaches=self.approaches, results=results)

        for index, result in enumerate(gathered_results):
            colors = ['r' if y < 0 else 'g' for y in result]
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.bar(range(len(results)), result, width=1.0, edgecolor='black', bottom=1,
                   color=colors)
            ax.hlines(1, -1, len(types) + 1, linestyle='--')
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels(types, rotation=45)
            plt.title(f'{self.approaches[index].__name__} results', fontsize=16)
            plt.xlabel('nodes', fontsize=14)
            plt.ylabel('changed_graph_metric/original_metric', fontsize=14)

            file_path = path_to_save_per_iter(root_path_to_save=self.path_to_save,
                                              file_name=f'{self.approaches[index].__name__}',
                                              extension='png',
                                              folder_name='results_per_approach')

            plt.savefig(file_path)
            self.log.message(f'Nodes Structural Analysis visualized results per approach were saved to {file_path}')

    @staticmethod
    def _get_unique_points(graph: OptGraph, nodes_degrees: list, result: list):
        """ Leaves one point with the same values along the x and y axes,
        and adds information about all other points with the same coordinates to the annotation of this point.
        It is done so that when visualization points annotations to points do not overlap each other"""

        points = []
        for i, (x, y, node) in enumerate(zip(nodes_degrees, result, graph.nodes)):
            is_already = False
            cur_point = {'x': x, 'y': y, 'annotation': f'{i}_{node}'}
            for point in points:
                if point['x'] == x and point['y'] == y:
                    point['annotation'] += f', {i}_{node}'
                    is_already = True
                    break
            if not is_already:
                points.append(cur_point)
        return points

    def _visualize_degree_correlation(self, graph: OptGraph, results: dict):
        nodes_degrees = get_nodes_degrees(graph)
        gathered_results = extract_result_values(approaches=self.approaches, results=results)
        for index, result in enumerate(gathered_results):
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title('Degree correlation', fontsize=16)
            ax.set_xlabel('nodes degrees', fontsize=14)
            ax.set_ylabel('metric if this node was dropped / metric of source graph - 1', fontsize=14)
            ax.scatter(nodes_degrees, result)

            points = self._get_unique_points(graph, nodes_degrees, result)

            for point in points:
                ax.annotate(point['annotation'], xy=(point['x'] + (max(nodes_degrees) - min(nodes_degrees)) / 100,
                                                     point['y'] + (max(result) - min(result)) / 100))

            file_path = path_to_save_per_iter(root_path_to_save=self.path_to_save,
                                              file_name=f'{self.approaches[index].__name__}_degree_correlation',
                                              extension='png',
                                              folder_name='degree_correlation')

            plt.savefig(file_path)
            self.log.message(f'Nodes degree correlation visualized results were saved to {file_path}')


def get_nodes_degrees(graph: OptGraph) -> Sequence[int]:
    """Nodes degree as the number of edges the node has:
        ``degree = #input_edges + #out_edges``

    Returns:
        nodes degrees ordered according to the nx_graph representation of this graph
    """
    graph, _ = graph_structure_as_nx_graph(graph)
    index_degree_pairs = graph.degree
    node_degrees = [node_degree[1] for node_degree in index_degree_pairs]
    return node_degrees


def path_to_save_per_iter(root_path_to_save, file_name: str, extension: str, folder_name: str = None):

    if folder_name:
        folder_path = os.path.join(root_path_to_save, folder_name)
    else:
        folder_path = root_path_to_save

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    idx = 0
    while os.path.exists(join(folder_path, f'{file_name}_iter_{idx}.{extension}')):
        idx += 1
    file_path = join(folder_path,
                     f'{file_name}_iter_{idx}.{extension}')
    return file_path
