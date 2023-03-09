from __future__ import annotations

import datetime
import os
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union
from uuid import uuid4

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import ArrowStyle
from pyvis.network import Network
from seaborn import color_palette

from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.dag.graph_utils import distance_to_primary_level
from golem.core.log import default_log
from golem.core.paths import default_data_dir

if TYPE_CHECKING:
    from golem.core.dag.graph import Graph
    from golem.core.dag.graph_node import GraphNode
    from golem.core.optimisers.graph import OptGraph

    GraphType = Union[Graph, OptGraph]
    GraphConvertType = Callable[[GraphType], Tuple[nx.DiGraph, Dict[uuid4, GraphNode]]]

PathType = Union[os.PathLike, str]

MatplotlibColorType = Union[str, Sequence[float]]
LabelsColorMapType = Dict[str, MatplotlibColorType]
NodeColorFunctionType = Callable[[Iterable[str]], LabelsColorMapType]
NodeColorType = Union[MatplotlibColorType, LabelsColorMapType, NodeColorFunctionType]


class GraphVisualizer:
    def __init__(self, graph: GraphType, visuals_params: Optional[Dict[str, Any]] = None,
                 to_nx_convert_func: GraphConvertType = graph_structure_as_nx_graph):
        visuals_params = visuals_params or {}
        default_visuals_params = dict(
            engine='matplotlib',
            dpi=100,
            node_color=self._get_colors_by_labels,
            node_size_scale=1.0,
            font_size_scale=1.0,
            edge_curvature_scale=1.0,
            figure_size=(7, 7)
        )
        default_visuals_params.update(visuals_params)
        self.visuals_params = default_visuals_params
        self.to_nx_convert_func = to_nx_convert_func
        self._update_graph(graph)
        self.log = default_log(self)

    def _update_graph(self, graph: GraphType):
        self.graph = graph
        self.nx_graph, self.nodes_dict = self.to_nx_convert_func(self.graph)

    def visualise(self, save_path: Optional[PathType] = None, engine: Optional[str] = None,
                  node_color: Optional[NodeColorType] = None, dpi: Optional[int] = None,
                  node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
                  edge_curvature_scale: Optional[float] = None, figure_size: Optional[Tuple[int, int]] = None):
        engine = engine or self._get_predefined_value('engine')

        if not self.graph.nodes:
            raise ValueError('Empty graph can not be visualized.')

        if engine == 'matplotlib':
            self._draw_with_networkx(save_path, node_color, dpi, node_size_scale, font_size_scale, edge_curvature_scale,
                                     figure_size=figure_size)
        elif engine == 'pyvis':
            self._draw_with_pyvis(save_path, node_color)
        elif engine == 'graphviz':
            self._draw_with_graphviz(save_path, node_color, dpi)
        else:
            raise NotImplementedError(f'Unexpected visualization engine: {engine}. '
                                      'Possible values: matplotlib, pyvis, graphviz.')

    def draw_nx_dag(self, ax: Optional[plt.Axes] = None, node_color: Optional[NodeColorType] = None,
                    node_size_scale: Optional[float] = None, font_size_scale: Optional[float] = None,
                    edge_curvature_scale: Optional[float] = None):

        node_color = node_color or self._get_predefined_value('node_color')
        node_size_scale = node_size_scale or self._get_predefined_value('node_size_scale')
        font_size_scale = font_size_scale or self._get_predefined_value('font_size_scale')
        edge_curvature_scale = (edge_curvature_scale if edge_curvature_scale is not None
                                else self._get_predefined_value('edge_curvature_scale'))

        nx_graph, nodes = self.nx_graph, self.nodes_dict

        if ax is None:
            ax = plt.gca()

        # Define colors
        if callable(node_color):
            node_color = node_color([str(node) for node in nodes.values()])
        if isinstance(node_color, dict):
            node_color = [node_color.get(str(node), node_color.get(None)) for node in nodes.values()]
        else:
            node_color = [node_color for _ in nodes]
        # Get nodes positions
        (pos,
         longest_x_sequence,
         longest_y_sequence) = GraphVisualizer._get_hierarchy_pos(nx_graph, nodes)
        longest_nodes_sequence_size = max(longest_x_sequence, longest_y_sequence)
        node_size = GraphVisualizer._get_scaled_node_size(longest_nodes_sequence_size, node_size_scale)

        if longest_nodes_sequence_size > 6:
            self._draw_nx_graph_with_legend(ax, pos, nodes, node_color, node_size, font_size_scale)
        else:
            self._draw_nx_graph_with_labels(ax, pos, nodes, node_color, node_size, font_size_scale,
                                            longest_nodes_sequence_size)
        self._draw_nx_curved_edges(ax, pos, node_size, edge_curvature_scale)

    def _get_predefined_value(self, param: str):
        if param not in self.visuals_params:
            self.log.warning(f'No default param found: {param}.')
        return self.visuals_params.get(param)

    def _draw_with_networkx(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None,
                            dpi: Optional[int] = None, node_size_scale: Optional[float] = None,
                            font_size_scale: Optional[float] = None, edge_curvature_scale: Optional[float] = None,
                            figure_size: Optional[Tuple[int, int]] = None):
        save_path = save_path or self._get_predefined_value('save_path')
        node_color = node_color or self._get_predefined_value('node_color')
        dpi = dpi or self._get_predefined_value('dpi')
        figure_size = figure_size or self._get_predefined_value('figure_size')

        ax = GraphVisualizer._setup_matplotlib_figure(figure_size, dpi)
        self.draw_nx_dag(ax, node_color, node_size_scale, font_size_scale, edge_curvature_scale)
        GraphVisualizer._rescale_matplotlib_figure(ax)
        if not save_path:
            plt.show()
        else:
            plt.savefig(save_path, dpi=dpi)
            plt.close()

    def _draw_with_pyvis(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None):
        save_path = save_path or self._get_predefined_value('save_path')
        node_color = node_color or self._get_predefined_value('node_color')

        net = Network('500px', '1000px', directed=True)
        nx_graph, nodes = self.nx_graph, self.nodes_dict
        node_color = self._define_colors(node_color, nodes)
        for n, data in nx_graph.nodes(data=True):
            operation = nodes[n]
            label = str(operation)
            data['n_id'] = str(n)
            data['label'] = label.replace('_', ' ')
            params = operation.content.get('params')
            if isinstance(params, dict):
                params = str(params)[1:-1]
            data['title'] = params
            data['level'] = distance_to_primary_level(operation)
            data['color'] = to_hex(node_color.get(label, node_color.get(None)))
            data['font'] = '20px'
            data['labelHighlightBold'] = True

        for _, data in nx_graph.nodes(data=True):
            net.add_node(**data)
        for u, v in nx_graph.edges:
            net.add_edge(str(u), str(v))

        if save_path:
            net.save_graph(str(save_path))
            return
        save_path = Path(default_data_dir(), 'graph_plots', str(uuid4()) + '.html')
        save_path.parent.mkdir(exist_ok=True)
        net.show(str(save_path))
        remove_old_files_from_dir(save_path.parent)

    def _draw_with_graphviz(self, save_path: Optional[PathType] = None, node_color: Optional[NodeColorType] = None,
                            dpi: Optional[int] = None):
        save_path = save_path or self._get_predefined_value('save_path')
        node_color = node_color or self._get_predefined_value('node_color')
        dpi = dpi or self._get_predefined_value('dpi')

        nx_graph, nodes = self.nx_graph, self.nodes_dict
        node_color = self._define_colors(node_color, nodes)
        for n, data in nx_graph.nodes(data=True):
            label = str(nodes[n])
            data['label'] = label.replace('_', ' ')
            data['color'] = to_hex(node_color.get(label, node_color.get(None)))

        gv_graph = nx.nx_agraph.to_agraph(nx_graph)
        kwargs = {'prog': 'dot', 'args': f'-Gnodesep=0.5 -Gdpi={dpi} -Grankdir="LR"'}

        if save_path:
            gv_graph.draw(save_path, **kwargs)
        else:
            save_path = Path(default_data_dir(), 'graph_plots', str(uuid4()) + '.png')
            save_path.parent.mkdir(exist_ok=True)
            gv_graph.draw(save_path, **kwargs)

            img = plt.imread(str(save_path))
            plt.imshow(img)
            plt.gca().axis('off')
            plt.gcf().set_dpi(dpi)
            plt.tight_layout()
            plt.show()
            remove_old_files_from_dir(save_path.parent)

    @staticmethod
    def _get_scaled_node_size(nodes_amount: int, size_scale: float) -> float:
        min_size = 150
        max_size = 12000
        size = max(max_size * (1 - np.log10(nodes_amount)), min_size)
        return size * size_scale

    @staticmethod
    def _get_scaled_font_size(nodes_amount: int, size_scale: float) -> float:
        min_size = 11
        max_size = 25
        size = max(max_size * (1 - np.log10(nodes_amount)), min_size)
        return size * size_scale

    @staticmethod
    def _get_colors_by_labels(labels: Iterable[str]) -> LabelsColorMapType:
        unique_labels = list(set(labels))
        palette = color_palette('tab10', len(unique_labels))
        return {label: palette[unique_labels.index(label)] for label in labels}

    @staticmethod
    def _define_colors(node_color, nodes):
        if callable(node_color):
            colors = node_color([str(node) for node in nodes.values()])
        elif isinstance(node_color, dict):
            colors = node_color
        else:
            colors = {str(node): node_color for node in nodes.values()}
        return colors

    @staticmethod
    def _setup_matplotlib_figure(figure_size: Tuple[float, float], dpi: int) -> plt.Axes:
        fig, ax = plt.subplots(figsize=figure_size)
        fig.set_dpi(dpi)
        return ax

    @staticmethod
    def _rescale_matplotlib_figure(ax):
        """Rescale the figure for all nodes to fit in."""

        x_1, x_2 = ax.get_xlim()
        y_1, y_2 = ax.get_ylim()
        offset = 0.2
        x_offset = x_2 * offset
        y_offset = y_2 * offset
        ax.set_xlim(x_1 - x_offset, x_2 + x_offset)
        ax.set_ylim(y_1 - y_offset, y_2 + y_offset)
        ax.axis('off')
        plt.tight_layout()

    def _draw_nx_curved_edges(self, ax, pos, node_size, edge_curvature_scale):
        nx_graph = self.nx_graph
        # The ongoing section defines curvature for all edges.
        #   This is 'connection style' for an edge that does not intersect any nodes.
        connection_style = 'arc3'
        #   This is 'connection style' template for an edge that is too close to any node and must bend around it.
        #   The curvature value is defined individually for each edge.
        connection_style_curved_template = connection_style + ',rad={}'
        default_edge_curvature = 0.3
        #   The minimum distance from a node to an edge on which the edge must bend around the node.
        node_distance_gap = 0.15
        for u, v, e in nx_graph.edges(data=True):
            e['connectionstyle'] = connection_style
            p_1, p_2 = np.array(pos[u]), np.array(pos[v])
            p_1_2 = p_2 - p_1
            p_1_2_length = np.linalg.norm(p_1_2)
            # Finding the closest node to the edge.
            min_distance_found = node_distance_gap * 2  # It just must be bigger than the gap.
            closest_node_id = None
            for node_id in nx_graph.nodes:
                if node_id in (u, v):
                    continue  # The node is adjacent to the edge.
                p_3 = np.array(pos[node_id])
                distance_to_node = abs(np.cross(p_1_2, p_3 - p_1)) / p_1_2_length
                if (distance_to_node > min(node_distance_gap, min_distance_found)  # The node is too far.
                        or ((p_3 - p_1) @ p_1_2) < 0  # There's no perpendicular from the node to the edge.
                        or ((p_3 - p_2) @ -p_1_2) < 0):
                    continue
                min_distance_found = distance_to_node
                closest_node_id = node_id

            if closest_node_id is None:
                continue  # There's no node to bend around.
            # Finally, define the edge's curvature based on the closest node position.
            p_3 = np.array(pos[closest_node_id])
            p_1_3 = p_3 - p_1
            curvature_strength = default_edge_curvature * edge_curvature_scale
            # 'alpha' denotes the angle between the abscissa and the edge.
            cos_alpha = p_1_2[0] / p_1_2_length
            sin_alpha = np.sqrt(1 - cos_alpha ** 2) * (-1) ** (p_1_2[1] < 0)
            # The closest node is placed as if the edge matched the abscissa.
            # Then, its ordinate shows on which side of the edge it is, "on the left" or "on the right".
            rotation_matrix = np.array([[cos_alpha, sin_alpha], [-sin_alpha, cos_alpha]])
            p_1_3_rotated = rotation_matrix @ p_1_3
            curvature_direction = (-1) ** (p_1_3_rotated[1] < 0)  # +1 is a "cup" \/, -1 is a "cat" /\.
            edge_curvature = curvature_direction * curvature_strength
            e['connectionstyle'] = connection_style_curved_template.format(edge_curvature)
            # Define edge center position for labels.
            edge_center_position = np.mean([p_1, p_2], axis=0)
            edge_curvature_shift = np.linalg.inv(rotation_matrix) @ [0, -1 * edge_curvature / 4]
            edge_center_position += edge_curvature_shift
            e['edge_center_position'] = edge_center_position
        # Draw the graph's edges.
        arrow_style = ArrowStyle('Simple', head_length=1.5, head_width=0.8)
        for u, v, e in nx_graph.edges(data=True):
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], node_size=node_size, ax=ax, arrowsize=10,
                                   arrowstyle=arrow_style, connectionstyle=e['connectionstyle'])

    def _draw_nx_graph_with_labels(self, ax, pos, nodes, node_color, node_size, font_size_scale,
                                   longest_nodes_sequence_size):
        # Draw the graph's nodes.
        nx.draw_networkx_nodes(self.nx_graph, pos, node_size=node_size, ax=ax, node_color='w', linewidths=3,
                               edgecolors=node_color)
        # Draw the graph's node labels.
        node_labels = {node_id: str(node) for node_id, node in nodes.items()}
        font_size = GraphVisualizer._get_scaled_font_size(longest_nodes_sequence_size, font_size_scale)
        for node, (x, y) in pos.items():
            text = '\n'.join(wrap(node_labels[node].replace('_', ' ').replace('-', ' '), 10))
            ax.text(x, y, text,
                    ha='center', va='center',
                    fontsize=font_size,
                    bbox=dict(alpha=0.9, color='w', boxstyle='round'))

    def _draw_nx_graph_with_legend(self, ax, pos, nodes, node_color, node_size, font_size_scale):
        nx_graph = self.nx_graph
        markers = 'os^>v<dph8'
        label_markers = {}
        labels_added = set()
        color_counts = {}
        for node_num, (node_id, node) in enumerate(nodes.items()):
            label = str(node)
            color = node_color[node_num]
            color = to_hex(color, keep_alpha=True)  # Convert the color to a hashable type.
            marker = label_markers.get(label)
            if marker is None:
                color_count = color_counts.get(color, 0)
                if color_count > len(markers) - 1:
                    self.log.warning(f'Too much node labels derive the same color: {color}. The markers may repeat.\n'
                                     '\tSpecify the parameter "node_color" to set distinct colors.')
                    color_count = color_count % len(markers)
                marker = markers[color_count]
                label_markers[label] = marker
                color_counts[color] = color_count + 1
            nx.draw_networkx_nodes(nx_graph, pos, [node_id], ax=ax, node_color=[color], node_size=node_size,
                                   node_shape=marker)
            if label in labels_added:
                continue
            ax.plot([], [], marker=marker, linestyle='None', color=color, label=label)
            labels_added.add(label)
        ax.legend(prop={'size': round(20 * font_size_scale)})

    @staticmethod
    def _get_hierarchy_pos(nx_graph: nx.DiGraph, nodes: Dict) -> Tuple[Dict[Any, Tuple[float, float]], int, int]:
        """By default, returns 'networkx.multipartite_layout' positions based on 'hierarchy_level`
        from node data - the property must be set beforehand.
        :param graph: the graph.
        """
        for node_id, node_data in nx_graph.nodes(data=True):
            node_data['hierarchy_level'] = distance_to_primary_level(nodes[node_id])

        longest_path = nx.dag_longest_path(nx_graph, weight=None)
        longest_x_sequence = len(longest_path)

        pos = nx.multipartite_layout(nx_graph, subset_key='hierarchy_level')

        y_level_nodes_count = {}
        longest_y_sequence = 1
        for x, _ in pos.values():
            y_level_nodes_count[x] = y_level_nodes_count.get(x, 0) + 1
            nodes_on_level = y_level_nodes_count[x]
            if nodes_on_level > longest_y_sequence:
                longest_y_sequence = nodes_on_level

        return pos, longest_x_sequence, longest_y_sequence


def remove_old_files_from_dir(dir_: Path, time_interval=datetime.timedelta(minutes=10)):
    for path in dir_.iterdir():
        if datetime.datetime.now() - datetime.datetime.fromtimestamp(path.stat().st_ctime) > time_interval:
            path.unlink()
