from datetime import datetime
from itertools import chain, product
from typing import Tuple, Optional, Sequence, Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm

from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.graph_viz import GraphVisualizer


def fitness_to_stats(history: OptHistory,
                     target_metric_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    all_metrics = history.historical_fitness
    if history.objective.is_multi_objective:
        quality = [[metrics[target_metric_index] for metrics in pop]
                   for pop in all_metrics]
    else:
        quality = all_metrics

    quality = np.array(quality)
    if quality.shape[1] == 1:
        # reshape array to get nontrivial samples for computing mean/std
        sample_size = 20
        shape = (len(quality) // sample_size, sample_size)
        num_elements = shape[0] * shape[1]
        quality = quality.squeeze()[:num_elements].reshape(shape)

    mean = np.mean(quality, axis=1)
    std = np.std(quality, axis=1)
    return mean, std


def plot_fitness_comparison(histories: Sequence[OptHistory],
                            target_metric_index: int = 0,
                            titles: Optional[Sequence[str]] = None,
                            total_minutes: Optional[float] = None,
                            ):
    stats = [fitness_to_stats(h, target_metric_index) for h in histories]

    if not titles:
        titles = [f'history_{i}' for i in range(len(histories))]
    common_len = min(len(mean) for mean, std in stats)
    xs = np.arange(0, common_len) if not total_minutes else np.linspace(0, total_minutes, common_len)

    fig, ax = plt.subplots()
    ax.set_title('Historical fitness comparison')
    ax.set_xlabel('minutes')
    ax.set_ylabel('fitness')
    ax.legend()
    fig.tight_layout()

    def resample(arr):
        old_indices = np.arange(0, len(arr))
        new_indices = np.linspace(0, len(arr), common_len)
        return np.interp(new_indices, old_indices, arr)

    for title, (mean, std) in zip(titles, stats):
        mean = resample(mean)
        std = resample(std)

        ax.plot(xs, mean, '-', label=title)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.25)

    return fig, ax


def plot_nx_graph(g: nx.DiGraph, ax: plt.Axes = None):
    adapter = BaseNetworkxAdapter()
    GraphVisualizer.draw_nx_dag(adapter.adapt(g), ax,
                                node_size_scale=0.2, font_size_scale=0.25,
                                edge_curvature_scale=0.5)


def draw_graphs_subplots(*graphs: nx.Graph,
                         titles: Optional[Sequence[str]] = None,
                         draw_fn=nx.draw_kamada_kawai,
                         size=10):
    graphs = [graphs] if not isinstance(graphs, Iterable) else graphs
    titles = [f'Graph #{i+1}' for i in range(len(graphs))] if not titles else titles
    # Setup subplots
    ncols = int(np.ceil(np.sqrt(len(graphs))))
    nrows = len(graphs) // ncols
    aspect = nrows / ncols
    figsize = (size, int(size * aspect))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = np.atleast_2d(axs)
    # Draw graphs
    for title, ax, graph in zip(titles, chain(*axs), graphs):
        colors, labeldict, legend_handles = _get_node_colors_and_labels(graph)
        draw_fn(graph, ax=ax, arrows=True,
                node_color=colors, with_labels=True, labels=labeldict)
        ax.set_title(title)
    fig.legend(handles=legend_handles)
    plt.show()


def _get_node_colors_and_labels(graph: nx.Graph, cmap_name='viridis'):
    degrees = dict(graph.degree())
    if isinstance(graph, nx.DiGraph):
        roots = {n for n, d in graph.out_degree() if d == 0}
        sources = {n for n, d in graph.in_degree() if d == 0}
    else:
        roots = {max(*graph.nodes(), key=lambda n: graph.degree[n])}
        sources = {min(*graph.nodes(), key=lambda n: graph.degree[n])}

    max_degree = max(degrees.values())
    root_cm = max_degree
    src_cm = 0
    colormap = cm.get_cmap(cmap_name, max_degree + 1)
    colors = []
    labels = {}

    for node, data in graph.nodes(data=True):
        # Determine color of the node:
        # if node is root -- use special 'min' color
        # if node is source -- use special 'max' color
        # else use node color according to colormap and node degree
        if node in roots:
            color = colormap(root_cm)
        elif node in sources:
            color = colormap(src_cm)
        else:
            color = colormap(graph.degree(node))
        colors.append(color)

        # Get node label
        label = data.get('name') or str(node)
        labels[node] = label

    # Construct legend handles
    root_legend = mpatches.Patch(color=colormap(root_cm), label='Root')
    degree_legend = mpatches.Patch(color=colormap(max_degree // 2 + 1), label='Node Degree')
    source_legend = mpatches.Patch(color=colormap(src_cm), label='Source')
    handles = [root_legend, degree_legend, source_legend]

    return colors, labels, handles
