from datetime import datetime, timedelta
from functools import partial
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Sequence, Type, Callable, Optional, List

import networkx as nx
import numpy as np

from examples.synthetic_graph_evolution.generators import generate_labeled_graph, graph_kinds
from examples.synthetic_graph_evolution.utils import draw_graphs_subplots
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.optimizer import GraphOptimizer
from golem.metrics.edit_distance import get_edit_dist_metric, matrix_edit_dist
from golem.metrics.graph_metrics import \
    spectral_dist, size_diff, degree_distance_kernel, degree_distance, nxgraph_stats


def get_all_quality_metrics(target_graph):
    quality_metrics = {
        'edit_distance': get_edit_dist_metric(target_graph),
        'matrix_edit_dist': partial(matrix_edit_dist, target_graph),
        'sp_adj': partial(spectral_dist, target_graph, kind='adjacency'),
        'sp_lapl': partial(spectral_dist, target_graph, kind='laplacian'),
        'sp_lapl_norm': partial(spectral_dist, target_graph, kind='laplacian_norm'),
        'graph_size': partial(size_diff, target_graph),
        'degree_dist_mmd': partial(degree_distance_kernel, target_graph),
        'degree_dist': partial(degree_distance, target_graph),
    }
    return quality_metrics


def run_experiments(optimizer_setup: Callable,
                    optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                    node_types: Optional[Sequence[str]] = None,
                    graph_names: Sequence[str] = graph_kinds,
                    graph_sizes: Sequence[int] = (30, 100, 300),
                    num_trials: int = 1,
                    trial_timeout: Optional[int] = None,
                    trial_iterations: Optional[int] = None,
                    visualize: bool = False,
                    ):
    log = StringIO()
    if not node_types:
        node_types = ['X']
    for graph_name, num_nodes in product(graph_names, graph_sizes):
        experiment_id = f'Experiment [graph={graph_name} graph_size={num_nodes}]'
        file_name = f'{optimizer_cls.__name__[:-9]}_{graph_name}_n{num_nodes}_iter{trial_iterations}'
        trial_results = []
        for i in range(num_trials):
            start_time = datetime.now()
            print(f'\nTrial #{i} of {experiment_id} started at {start_time}', file=log)

            # Generate random target graph and run the optimizer
            target_graph = generate_labeled_graph(graph_name, num_nodes, node_types)
            target_graph = target_graph.reverse()
            # Run optimizer setup
            optimizer, objective = optimizer_setup(target_graph,
                                                   optimizer_cls=optimizer_cls,
                                                   node_types=node_types,
                                                   timeout=timedelta(minutes=trial_timeout) if trial_timeout else None,
                                                   num_iterations=trial_iterations)
            found_graphs = optimizer.optimise(objective)
            found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
            history = optimizer.history

            trial_results.extend(history.final_choices)
            found_nx_graph = BaseNetworkxAdapter().restore(found_graph)

            duration = datetime.now() - start_time
            print(f'Trial #{i} finished, spent time: {duration}', file=log)
            print('target graph stats: ', nxgraph_stats(target_graph), file=log)
            print('found graph stats: ', nxgraph_stats(found_nx_graph), file=log)
            if visualize:
                draw_graphs_subplots(target_graph, found_nx_graph,
                                     titles=['Target Graph', 'Found Graph'], show=False)
                diversity_filename = f'./results/diversity_hist_{graph_name}_n{num_nodes}.gif'
                history.show.diversity_population(save_path=diversity_filename)
                history.show.diversity_line(show=False)
                history.show.fitness_line()
            result_dir = Path('results') / file_name
            result_dir.mkdir(parents=True, exist_ok=True)
            history.save(result_dir / f'history_trial_{i}.json')

        # Compute mean & std for metrics of trials
        ff = objective.format_fitness
        trial_metrics = np.array([ind.fitness.values for ind in trial_results])
        trial_metrics_mean = trial_metrics.mean(axis=0)
        trial_metrics_std = trial_metrics.std(axis=0)
        print(f'{experiment_id} finished with metrics:\n'
              f'mean={ff(trial_metrics_mean)}\n'
              f' std={ff(trial_metrics_std)}',
              file=log)
        print(log.getvalue())
    return log.getvalue()


def run_trial(target_graph: nx.DiGraph,
              optimizer_setup: Callable,
              optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
              timeout: Optional[timedelta] = None,
              num_iterations: Optional[int] = None,
              node_types: Optional[List[str]] = None):
    optimizer, objective = optimizer_setup(target_graph,
                                           optimizer_cls=optimizer_cls,
                                           timeout=timeout,
                                           node_types=node_types,
                                           num_iterations=num_iterations)
    found_graphs = optimizer.optimise(objective)
    found_graph = found_graphs[0] if isinstance(found_graphs, Sequence) else found_graphs
    history = optimizer.history
    return found_graph, history
