import json
import os
from copy import deepcopy
from typing import Callable, Union, Any, Optional, Tuple, Dict, List

import matplotlib.pyplot as plt
import networkx as nx

from golem.core.dag.graph import Graph
from golem.core.optimisers.graph import OptGraph
from golem.structural_analysis.graph_viz_temporary import NodeColorType, GraphVisualizer
from golem.structural_analysis.pipeline_sa.sa_approaches_repository import StructuralAnalysisApproachesRepository

GraphType = Union[Graph, OptGraph]

SA_ABBREVIATIONS = {
    'SubtreeDeletionAnalyze': 'SD',
    'NodeDeletionAnalyze': 'ND',
    'NodeReplaceOperationAnalyze': 'NR',
    'EdgeDeletionAnalyze': 'ED',
    'EdgeReplaceOperationAnalyze': 'ER'
}


def draw_nx_dag(graph: GraphType,
                save_path: str,
                json_path: str,
                metrics_names: List[str] = None,
                node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
                node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
                edge_curvature_scale: Optional[float] = None, figure_size: Optional[Tuple[int, int]] = None):
    """
    Visualise graph changes per iteration during SA

    Args:
        graph: graph to visualise
        save_path: path where to save pics
        json_path: path to json with SA changes
        metrics_names: metrics names
        node_color: node color
        dpi: dpi
        node_size_scale: node size
        font_size_scale: font size
        edge_curvature_scale: edge curvature
        figure_size: figure_size
    """
    SAPipelineVisualizer(graph).draw_with_sa(save_path=save_path, json_path=json_path,
                                             metrics_names=metrics_names,
                                             node_color=node_color, dpi=dpi,
                                             node_size_scale=node_size_scale,
                                             font_size_scale=font_size_scale,
                                             edge_curvature_scale=edge_curvature_scale,
                                             figure_size=figure_size)


class SAGraphVisualizer(GraphVisualizer):
    def __init__(self, graph: GraphType, visuals_params: Optional[Dict[str, Any]] = None):
        visuals_params = visuals_params if visuals_params is not None else {}
        visuals_params['dpi'] = visuals_params.get('dpi') or 300
        super().__init__(graph, visuals_params)

    def draw_with_sa(self,
                     save_path: str,
                     json_path: str,
                     metrics_names: List[str] = None,
                     node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
                     node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
                     edge_curvature_scale: Optional[float] = None, figure_size: Optional[Tuple[int, int]] = None):
        node_color = node_color or self._get_predefined_value('node_color')
        dpi = dpi or self._get_predefined_value('dpi')
        node_size_scale = node_size_scale or self._get_predefined_value('node_size_scale')
        font_size_scale = font_size_scale or self._get_predefined_value('font_size_scale')
        edge_curvature_scale = edge_curvature_scale or self._get_predefined_value('edge_curvature_scale')
        figure_size = figure_size or self._get_predefined_value('figure_size')

        analysis_results = json.load(open(json_path))
        num_iterations = SAGraphVisualizer.get_num_iterations(json_path)

        for iter in range(num_iterations):
            ax = GraphVisualizer._setup_matplotlib_figure(figure_size, dpi)

            graph = self.graph
            nx_graph, nodes = self.nx_graph, self.nodes_dict
            self.draw_nx_dag(ax, node_color,
                             node_size_scale, font_size_scale,
                             edge_curvature_scale)

            labels, worst_approach_name, entity_to_change = \
                self.get_labels_for_iteration(json_file_path=json_path, iter=iter,
                                              metrics_names=metrics_names)
            nodes_labels = labels['nodes_labels']
            edges_labels = labels['edges_labels']
            (pos,
             longest_x_sequence,
             longest_y_sequence) = GraphVisualizer._get_hierarchy_pos(nx_graph, nodes)
            longest_sequence = max(longest_x_sequence, longest_y_sequence)

            self.set_labels(ax, pos, nodes_labels,
                            edges_labels, longest_sequence,
                            longest_y_sequence, font_size_scale)
            SAGraphVisualizer._rescale_matplotlib_figure(ax)

            files = os.listdir(save_path)
            max_iter = -1
            for file in files:
                if 'viz_' in file:
                    iter_num = int(file.split('_')[1])
                    max_iter = iter_num if iter_num > max_iter else max_iter
            cur_save_path = os.path.join(save_path, f'viz_{max_iter + 1}_iter.png')
            plt.savefig(cur_save_path, dpi=dpi)
            print(f'Visualization was saved to: {cur_save_path}')
            plt.close()

            # Delete node/edge that has max score
            graph = SAGraphVisualizer._delete_useless_obj(analysis_results=analysis_results[iter], graph=graph,
                                                          worst_approach_name=worst_approach_name,
                                                          entity_to_change=entity_to_change)
            self._update_graph(graph)

    # Labels functions
    def set_labels(self, ax: plt.Axes, pos: Any, nodes_labels: dict, edges_labels: dict,
                   longest_sequence: int, longest_y_sequence: int, font_size_scale: float):
        """ Set labels with SA scores """
        nx_graph = self.nx_graph
        # Set labels for nodes
        labels_pos = deepcopy(pos)
        bias = SAGraphVisualizer.calculate_labels_bias(ax, longest_y_sequence)
        font_size = SAGraphVisualizer._get_scaled_font_size(longest_sequence, font_size_scale * 0.7)
        bbox = dict(alpha=0.9, color='w')
        for value in labels_pos.values():
            value[1] += bias
        nx.draw_networkx_labels(
            nx_graph, labels_pos,
            labels=nodes_labels,
            font_color='black',
            font_size=font_size,
            bbox=bbox
        )

        labels_pos_edges = deepcopy(pos)
        label_bias_y = 2 / 3 * bias
        if len(set([coord[1] for coord in pos.values()])) == 1 and len(list(pos.values())) > 2:
            for value in labels_pos_edges.values():
                value[1] += label_bias_y
        # Set labels for edges
        for u, v, e in nx_graph.edges(data=True):
            current_pos = labels_pos_edges
            if 'edge_center_position' in e:
                x, y = e['edge_center_position']
                plt.text(x, y, edges_labels[(u, v)], bbox=bbox, fontsize=font_size)
            else:
                nx.draw_networkx_edge_labels(
                    nx_graph, current_pos, {(u, v): edges_labels[(u, v)]},
                    label_pos=0.5, ax=ax,
                    font_color='black',
                    font_size=font_size,
                    rotate=False,
                    bbox=bbox
                )

    def get_labels_for_iteration(self, json_file_path: str,
                                 iter: int, metrics_names: List[str]):
        """ Get SA labels with scores from json """

        def _add_label_to_list(action: str, labels: dict, idx: int, actions_per_obj: list, graph: GraphType):
            action_abbreviation = SA_ABBREVIATIONS[action]
            scores = [round(score, 3) for score in actions_per_obj[idx][f'{action}']['loss']]
            if metrics_names:
                if 'Deletion' in action:
                    for j, metric in enumerate(metrics_names):
                        if j == 0:
                            labels[list(labels.keys())[idx]] += \
                                f"{action_abbreviation} " \
                                f"{metric}: {scores[j]}\n"
                        else:
                            labels[list(labels.keys())[idx]] += \
                                f"{metric}: {scores[j]}\n"
                if 'Replace' in action:
                    if 'Node' in action:
                        try:
                            for j, metric in enumerate(metrics_names):
                                if j == 0:
                                    labels[list(labels.keys())[idx]] += \
                                        f"{action_abbreviation} " \
                                        f"{metric}: {scores[j]} for " \
                                        f"{actions_per_obj[idx][f'{action}']['new_node_operation']}\n"
                                else:
                                    labels[list(labels.keys())[idx]] += \
                                        f"{metric}: {scores[j]} for " \
                                        f"{actions_per_obj[idx][f'{action}']['new_node_operation']}\n"

                        except KeyError:
                            pass
                    else:
                        new_edge_idxs = actions_per_obj[idx][f'{action}']['edge_node_idx_to_replace_to']
                        parent_idx = new_edge_idxs['parent_node_id']
                        child_idx = new_edge_idxs['child_node_id']
                        parent_node_name = graph.nodes[parent_idx]
                        child_node_name = graph.nodes[child_idx]
                        for j, metric in enumerate(metrics_names):
                            if j == 0:
                                labels[list(labels.keys())[idx]] += \
                                    f"{action_abbreviation} " \
                                    f"{metric}: {scores[j]} for edge " \
                                    f"{parent_node_name} -> {child_node_name}\n"
                            else:
                                labels[list(labels.keys())[idx]] += \
                                    f"{metric}: {scores[j]} for edge " \
                                    f"{parent_node_name} -> {child_node_name}\n"

            else:
                if len(scores) == 1:
                    scores = scores[0]
                if 'Deletion' in action:
                    labels[list(labels.keys())[idx]] += \
                        f"{action_abbreviation}: " \
                        f"{scores}\n"
                if 'Replace' in action:
                    if 'Node' in action:
                        try:
                            labels[list(labels.keys())[idx]] += \
                                f"{action_abbreviation} " \
                                f"{scores} for " \
                                f"{actions_per_obj[idx][f'{action}']['new_node_operation']}\n"
                        except KeyError:
                            pass
                    else:
                        new_edge_idxs = actions_per_obj[idx][f'{action}']['edge_node_idx_to_replace_to']
                        parent_idx = new_edge_idxs['parent_node_id']
                        child_idx = new_edge_idxs['child_node_id']
                        parent_node_name = graph.nodes[parent_idx]
                        child_node_name = graph.nodes[child_idx]
                        labels[list(labels.keys())[idx]] += \
                            f"{action_abbreviation} " \
                            f"{scores} for edge " \
                            f"{parent_node_name} -> {child_node_name}\n"

        graph = self.graph
        nx_graph = self.nx_graph
        labels_json = json.load(open(json_file_path))
        cur_iter_labels = labels_json[iter]

        nodes_labels = dict.fromkeys(nx_graph.nodes)
        for key in nodes_labels:
            nodes_labels[key] = ''
        edges_labels = dict.fromkeys(nx_graph.edges)
        for key in edges_labels:
            edges_labels[key] = ''

        max_score = 0
        worst_approach_name = ''
        entity = ''

        for approach in list(cur_iter_labels.keys()):
            actions_per_obj = list(cur_iter_labels[approach].values())
            for i in range(len(actions_per_obj)):
                actions_taken = list(actions_per_obj[i].keys())
                for action in actions_taken:
                    score = actions_per_obj[i][f'{action}']['loss']
                    if isinstance(score, list):
                        score = score[0]
                    if score >= max_score:
                        max_score = score
                        worst_approach_name = action
                        entity = list(cur_iter_labels[approach].keys())[i]
                    if approach == 'nodes_result':
                        _add_label_to_list(action=action, labels=nodes_labels,
                                           idx=i, actions_per_obj=actions_per_obj,
                                           graph=graph)
                    else:
                        _add_label_to_list(action=action, labels=edges_labels,
                                           idx=i, actions_per_obj=actions_per_obj,
                                           graph=graph)
        if max_score <= 1:
            worst_approach_name = None
        return {'nodes_labels': nodes_labels, 'edges_labels': edges_labels}, worst_approach_name, entity

    @staticmethod
    def calculate_labels_bias(ax: plt.Axes, longest_y_sequence: int):
        y_1, y_2 = ax.get_ylim()
        y_size = y_2 - y_1
        if longest_y_sequence == 1:
            bias_scale = 0.25  # Fits between the central line and the upper bound.
        else:
            bias_scale = 1 / longest_y_sequence / 2 * 0.9  # Fits between the narrowest horizontal rows.
        bias = y_size * bias_scale
        return bias

    @staticmethod
    def _rescale_matplotlib_figure(ax):
        """Rescale the figure for all nodes to fit in."""

        x_1, x_2 = ax.get_xlim()
        y_1, y_2 = ax.get_ylim()
        offset = 0.2
        x_offset = x_2 * offset
        y_offset = y_2 * offset
        ax.set_xlim(x_1 - x_offset, x_2 + x_offset)
        bias = SAGraphVisualizer.calculate_pos_bias(ax)
        ax.set_ylim(y_1 - y_offset, y_2 + y_offset + bias)
        ax.axis('off')
        plt.tight_layout()

    @staticmethod
    def calculate_pos_bias(ax):
        y_1, y_2 = ax.get_ylim()
        bias = (y_2 - y_1) / 5
        return bias

    # Other useful functions
    @staticmethod
    def get_num_iterations(json_file_path: str) -> int:
        return len(json.load(open(json_file_path)))

    @staticmethod
    def _delete_useless_obj(analysis_results: dict, graph: GraphType, worst_approach_name, entity_to_change):
        """ Deletes node/edge in graph that has the max score """
        if worst_approach_name is None:
            return graph
        postproc_method = StructuralAnalysisApproachesRepository().postproc_method_by_name(worst_approach_name)
        new_graph = postproc_method(results=analysis_results, pipeline=graph, entity=entity_to_change)
        return new_graph


class SAPipelineVisualizer(SAGraphVisualizer):
    def __init__(self, graph: GraphType, visuals_params: Optional[Dict[str, Any]] = None):
        visuals_params = visuals_params if visuals_params is not None else {}
        # TODO: get_colors_by_tags from fedot
        visuals_params['node_color'] = visuals_params.get('node_color') or get_colors_by_tags
        super().__init__(graph, visuals_params)
