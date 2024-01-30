import os
from datetime import timedelta
from functools import partial

import networkx as nx
from matplotlib import pyplot as plt, animation
from typing import List

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.utils import _get_node_colors_and_labels
from golem.core.adapter import BaseOptimizationAdapter
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.metrics.edit_distance import tree_edit_dist
from golem.visualisation.opt_viz_extra import OptHistoryExtraVisualizer


def run_graph_search(size=16, timeout=8, visualize=True):
    # Generate target graph that will be sought by optimizer
    node_types = ('a', 'b')
    target_graph = generate_labeled_graph('tree', size, node_labels=node_types)

    # Generate initial population with small tree graphs
    initial_graphs = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]
    # Setup objective: edit distance to target graph
    objective = Objective(partial(tree_edit_dist, target_graph))

    # Setup optimization parameters
    requirements = GraphRequirements(
        early_stopping_iterations=100,
        timeout=timedelta(minutes=timeout),
        n_jobs=-1,
    )
    gp_params = GPAlgorithmParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        max_pop_size=50,
        mutation_types=[MutationTypesEnum.single_add,
                        MutationTypesEnum.single_drop,
                        MutationTypesEnum.single_change],
        crossover_types=[CrossoverTypesEnum.subtree]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=BaseNetworkxAdapter(),  # Example works with NetworkX graphs
        rules_for_constraint=DEFAULT_DAG_RULES,  # We don't want cycles in the graph
        available_node_types=node_types  # Node types that can appear in graphs
    )
    all_parameters = (requirements, graph_gen_params, gp_params)

    # Build and run the optimizer
    optimiser = EvoGraphOptimizer(objective, initial_graphs, *all_parameters)
    found_graphs = optimiser.optimise(objective)
    print(found_graphs[0].descriptive_id)
    print(found_graphs[0].graph_description)

    if visualize:
        vis = OptHistoryExtraVisualizer(optimiser.history, r"C:\dev\aim\GOLEM\examples\synthetic_graph_evolution\data")
        vis.visualize_best_genealogical_path(graph_gen_params.adapter.adapt_func(tree_edit_dist),
                                             graph_gen_params.adapter.adapt(target_graph))
        # vis.visualise_history()
        # vis.pareto_gif_create()
        # vis.boxplots_gif_create()

        # optimiser.history.show.fitness_box()
        # optimiser.history.show.fitness_line()
        # optimiser.history.show.fitness_line_interactive()
        # optimiser.history.show.operations_kde()
        # optimiser.history.show.operations_animated_bar()
        # optimiser.history.show.diversity_line()
        # optimiser.history.show.diversity_population(save_path="diversity.mp4")
        # Restore the NetworkX graph back from internal Graph representation
        # animate_graph_evolution(target_graph, optimiser.history, graph_gen_params.adapter, "./")

    return found_graphs


# def animate_graph_evolution(target_graph: nx.Graph, evolution_history: OptHistory, adapter: BaseOptimizationAdapter, dir_to_save_gif: str):

if __name__ == '__main__':
    """
    In this example Optimizer is expected to find the target graph
    using Tree Edit Distance metric and a random tree (nx.random_tree) as target.
    The convergence can be seen from achieved metrics and visually from graph plots.
    """
    run_graph_search(visualize=True)
