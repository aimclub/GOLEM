from datetime import datetime
from itertools import chain
from typing import Tuple, Optional, Sequence, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx import gnp_random_graph

from golem.metrics.graph_metrics import spectral_dists_all
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
                         draw_fn=nx.draw_kamada_kawai,
                         size=10):
    graphs = [graphs] if not isinstance(graphs, Iterable) else graphs
    # Setup subplots
    ncols = int(np.ceil(np.sqrt(len(graphs))))
    nrows = len(graphs) // ncols
    aspect = nrows / ncols
    figsize = (size, int(size * aspect))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = [axs] if not isinstance(axs, Iterable) else axs
    # Draw graphs
    for ax, graph in zip(chain(*axs), graphs):
        colors, labeldict = _get_node_colors_and_labels(graph)
        draw_fn(graph, ax=ax, arrows=True,
                node_color=colors, with_labels=True, labels=labeldict)
    plt.show()


def _get_node_colors_and_labels(graph: nx.Graph):
    clr_cyan = '#2A788EFF'
    clr_yellow = '#FDE725FF'
    clr_green = '#7AD151FF'

    if isinstance(graph, nx.DiGraph):
        roots = {n for n, d in graph.out_degree() if d == 0}
        sources = {n for n, d in graph.in_degree() if d == 0}
    else:
        roots = {max(*graph.nodes(), key=lambda n: graph.degree[n])}
        sources = {min(*graph.nodes(), key=lambda n: graph.degree[n])}

    colors = []
    labels = {}
    for node, data in graph.nodes(data=True):
        if node in roots:
            color = clr_yellow
        elif node in sources:
            color = clr_green
        else:
            color = clr_cyan
        colors.append(color)
        label = data.get('name') or str(node)
        labels[node] = label
    return colors, labels


def measure_graphs(target_graph, graph, vis=False):
    start = datetime.now()
    print("Computing metric...")
    fitness = spectral_dists_all(target_graph, graph)
    fitness2 = spectral_dists_all(target_graph, graph, match_size=False)
    fitness3 = spectral_dists_all(target_graph, graph, k=10)
    end = datetime.now() - start
    print(f'metrics: {fitness}, computed for '
          f'size {len(target_graph.nodes)} in {end.seconds} sec.')
    print(f'metrics2: {fitness2}')
    print(f'metrics3: {fitness3}')

    if vis:
        # 2 subplots
        fig, axs = plt.subplots(nrows=1, ncols=2)
        for g, ax in zip((target_graph, graph), axs):
            plot_nx_graph(g, ax)

        plt.title(f'metrics: {fitness.values}')
        plt.show()


def try_random(n=30, it=1):
    graphs = []
    for i in range(it):
        for p in [0.05, 0.08, 0.15, 0.3]:
            g1 = gnp_random_graph(n, p)
            g2 = gnp_random_graph(n, p)
            graphs.append(g1)
            graphs.append(g2)
            measure_graphs(g1, g2, vis=False)
    draw_graphs_subplots(*graphs, size=12)


if __name__ == "__main__":
    try_random()
