from datetime import timedelta, datetime
from functools import partial
from itertools import product
from typing import Callable, Sequence, Optional, Dict

import networkx as nx
import numpy as np

from examples.synthetic_graph_evolution.graph_metrics import *
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.optimisers.optimization_parameters import OptimizationParameters, GraphRequirements
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.visualisation.opt_history.graphs_interactive import GraphsInteractive

NumNodes = int
DiGraphGenerator = Callable[[NumNodes], nx.DiGraph]


def nx_to_directed(graph: nx.Graph) -> nx.DiGraph:
    """Randomly chooses a direction for each edge."""
    dedges = set()
    digraph = nx.DiGraph()

    for node, data in graph.nodes(data=True):
        digraph.add_node(node, **data)

    for u, v, data in graph.edges.data():
        edge = (u, v)
        inv_edge = (v, u)
        if edge in dedges or inv_edge in dedges:
            continue

        if np.random.default_rng().random() > 0.5:
            digraph.add_edge(*edge, **data)
            dedges.add(edge)
        else:
            digraph.add_edge(*inv_edge, **data)
            dedges.add(inv_edge)
    return digraph


graph_generators: Dict[str, DiGraphGenerator] = {
    'star': lambda n: nx_to_directed(nx.star_graph(n)),
    'grid2d': lambda n: nx.grid_2d_graph(int(np.sqrt(n)), int(np.sqrt(n))),
    '2ring': lambda n: nx_to_directed(nx.circular_ladder_graph(n)),
    'hypercube': lambda n: nx_to_directed(nx.hypercube_graph(int(np.log2(n).round()))),
    'gnp': lambda n: nx.gnp_random_graph(n, p=0.15)
}


def get_all_quality_metrics(target_graph):
    quality_metrics = {
        'edit_distance': get_edit_dist_metric(target_graph),
        'matrix_edit_dist': partial(matrix_edit_dist, target_graph),
        'sp_adj': partial(spectral_dist, target_graph, kind='adjacency'),
        'sp_lapl': partial(spectral_dist, target_graph, kind='laplacian'),
        'sp_lapl_norm': partial(spectral_dist, target_graph, kind='laplacian_norm'),
        'graph_size': partial(size_diff, target_graph),
    }
    return quality_metrics


def run_experiments(graph_names: Sequence[str] = tuple(graph_generators.keys()),
                    graph_sizes: Sequence[int] = (30, 100, 300),
                    num_trials: int = 1,
                    trial_timeout: Optional[int] = None,
                    visualize: bool = False,
                    ):
    for graph_name, num_nodes in product(graph_names, graph_sizes):
        graph_generator = graph_generators[graph_name]
        for i in range(num_trials):
            start_time = datetime.now()
            print(f'Trial #{i} with graph={graph_name} graph_size={num_nodes} at time {start_time}')

            target_graph = graph_generator(num_nodes)
            found_graph, history = run_experiment(target_graph, num_nodes,
                                                  timeout=timedelta(minutes=trial_timeout))
            found_nx_graph = BaseNetworkxAdapter().restore(found_graph)

            duration = datetime.now() - start_time
            print(f'Trial #{i} finished, spent time: {duration}')
            print('target graph stats: ', nxgraph_stats(target_graph))
            print('found graph stats: ', nxgraph_stats(found_nx_graph))
            if visualize:
                # nx.draw(target_graph)
                nx.draw_kamada_kawai(target_graph, arrows=True)
                GraphsInteractive(history).visualize()
                history.show.fitness_line_interactive()


def run_experiment(target_graph: nx.DiGraph,
                   num_nodes: int = 50,
                   timeout: Optional[timedelta] = None):
    # Setup parameters
    requirements = GraphRequirements(
        max_arity=num_nodes,
        max_depth=num_nodes,
        early_stopping_timeout=5,
        early_stopping_iterations=1000,
        timeout=timeout,
        n_jobs=-1,
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.single_add,
            MutationTypesEnum.single_edge,
            MutationTypesEnum.single_drop,
        ]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes,],
    )

    # Setup objective that measures some graph-theoretic similarity measure
    objective = Objective(
        quality_metrics={
            'sp_adj': partial(spectral_dist, target_graph, kind='adjacency'),
            'sp_lapl': partial(spectral_dist, target_graph, kind='laplacian'),
        },
        complexity_metrics={
            'graph_size': partial(size_diff, target_graph),
        },
        is_multi_objective=True
    )
    # Generate simple initial population with single-node graphs
    initial_graphs = [OptGraph(OptNode(f'Node{i}')) for i in range(gp_params.pop_size)]

    # Run the optimizer
    optimiser = EvoGraphOptimizer(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    found_graphs = optimiser.optimise(objective)

    return found_graphs[0], optimiser.history


if __name__ == '__main__':
    # seed = 321
    # random.seed(seed)
    # np.random.seed(seed)

    run_experiments(['2ring', 'hypercube', 'gnp'],
                    graph_sizes=(10, 50,),
                    trial_timeout=10,
                    visualize=True)
