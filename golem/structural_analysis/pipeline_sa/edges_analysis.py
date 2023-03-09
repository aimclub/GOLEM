import json
from os.path import join
from typing import Optional, List, Type, Callable

from matplotlib import pyplot as plt
import multiprocessing

from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.pipeline_sa.edge_sa_approaches import EdgeAnalyzeApproach, EdgeAnalysis
from golem.structural_analysis.pipeline_sa.entities.edge import Edge
from golem.structural_analysis.pipeline_sa.nodes_analysis import path_to_save_per_iter
from golem.structural_analysis.pipeline_sa.postproc_methods import extract_result_values
from golem.structural_analysis.pipeline_sa.sa_approaches_repository import EDGE_REPLACEMENT
from golem.structural_analysis.pipeline_sa.sa_requirements import StructuralAnalysisRequirements


class EdgesAnalysis:
    """
    This class is for edges structural analysis within an OptGraph .
    It takes edges and approaches to be applied to chosen edges.
    To define which edges to analyze pass them to edges_to_analyze filed
    or all edges will be analyzed.

    :param pipeline: pipeline object to analyze
    :param objectives: list of objective functions for computing metric values
    :param approaches: methods applied to edges to modify the pipeline or analyze certain operations.\
    Default: [EdgeDeletionAnalyze, EdgeReplaceOperationAnalyze]
    :param edges_to_analyze: edges to analyze. Default: all edges
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, pipeline: OptGraph, objectives: List[Callable],
                 approaches: Optional[List[Type[EdgeAnalyzeApproach]]] = None,
                 requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None,
                 edges_to_analyze: List[Edge] = None):

        self.pipeline = pipeline
        self.objectives = objectives
        self.approaches = approaches
        self.requirements = \
            StructuralAnalysisRequirements() if requirements is None else requirements
        self.metric = self.requirements.metric
        self.log = default_log(self)
        self.path_to_save = \
            join(default_data_dir(), 'structural', 'edges_structural') if path_to_save is None else path_to_save

        if not edges_to_analyze:
            self.log.message('Edges to analyze are not defined. All edges will be analyzed.')
            self.edges_to_analyze = Edge.from_tuple(self.pipeline.get_edges())
        else:
            self.edges_to_analyze = edges_to_analyze

    def analyze(self, n_jobs: int = -1, timer: OptimisationTimer = None) -> dict:
        """
        Main method to run the analyze process for every edge.

        :return edges_results: dict with analysis result per Edge
        """

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        edges_results = dict()
        operation_types = []
        edges_to_replace_to = []
        edge_analysis = EdgeAnalysis(approaches=self.approaches,
                                     approaches_requirements=self.requirements,
                                     path_to_save=self.path_to_save)

        with multiprocessing.Pool(processes=n_jobs) as pool:
            edges_result = pool.starmap(edge_analysis.analyze,
                                        [[self.pipeline, edge, self.objectives, timer]
                                         for edge in self.edges_to_analyze])

            for i, edge in enumerate(self.edges_to_analyze):
                edges_results[f'parent_node id = {self.pipeline.nodes.index(edge.parent_node)}, '
                              f'child_node id = {self.pipeline.nodes.index(edge.child_node)}'] = edges_result[i]
                operation_types.append(f'{self.pipeline.nodes.index(edge.parent_node)}_{edge.parent_node.operation} '
                                       f'{self.pipeline.nodes.index(edge.child_node)}_{edge.child_node.operation}')

        if self.requirements.is_visualize:
            # get edges to replace to for visualization
            for edge_result in edges_result:
                if EDGE_REPLACEMENT in edge_result.keys():
                    edges_to_replace_to.append(edge_result[EDGE_REPLACEMENT]['edge_node_idx_to_replace_to'])

            self._visualize_result_per_approach(edges_results, operation_types, edges_to_replace_to)

        if self.requirements.is_save:
            self._save_results_to_json(edges_results)

        return edges_results

    def _save_results_to_json(self, result: dict):
        file_path = path_to_save_per_iter(root_path_to_save=self.path_to_save,
                                          file_name=f'{self.approaches[0].__name__}_results',
                                          extension='json',
                                          folder_name='json_results')

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(json.dumps(result, indent=4))

        self.log.message(f'Edges Structural Analysis results were saved to {file_path}')

    def _visualize_result_per_approach(self, results: dict, types: list, edges_idxs_to_replace_to: list):
        gathered_results = extract_result_values(approaches=self.approaches, results=results)

        for index, result in enumerate(gathered_results):
            colors = ['r' if y < 0 else 'g' for y in result]
            fig, ax = plt.subplots(figsize=(22, 15))
            ax.bar(range(len(results)), result, width=1.0, edgecolor='black', bottom=1,
                   color=colors)
            ax.hlines(1, -1, len(types) + 1, linestyle='--')
            ax.set_xticks(range(len(results)))

            if self.approaches[index].__name__ == EDGE_REPLACEMENT:
                nodes_to_replace_to = []
                for nodes in edges_idxs_to_replace_to:
                    parent_node_idx = nodes['parent_node_id']
                    child_node_idx = nodes['child_node_id']
                    nodes_to_replace_to.append(f'\nto\n'
                                               f'{parent_node_idx}_{self.pipeline.nodes[parent_node_idx].operation}_'
                                               f'{child_node_idx}_{self.pipeline.nodes[child_node_idx].operation}')
                types = list(map(lambda x, y: x + y, types, nodes_to_replace_to))

            ax.set_xticklabels(types, rotation=25)
            plt.title(f'{self.approaches[index].__name__} results', fontsize=18)
            plt.xlabel('parent node _ child node', fontsize=16)
            plt.ylabel('changed_pipeline_metric/original_metric', fontsize=16)

            file_path = path_to_save_per_iter(root_path_to_save=self.path_to_save,
                                              file_name=f'{self.approaches[index].__name__}',
                                              extension='png',
                                              folder_name='results_per_approach')

            plt.savefig(file_path)
            self.log.message(f'Edges Structural Analysis visualized results per approach were saved to {file_path}')
