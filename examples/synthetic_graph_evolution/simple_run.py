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


def run_graph_search(size=16, timeout=0.2, visualize=True):
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


    if visualize:
        # vis = OptHistoryExtraVisualizer(optimiser.history, r"C:\dev\aim\GOLEM\examples\synthetic_graph_evolution\data")
        # vis.visualise_history()
        optimiser.history.show()
        # Restore the NetworkX graph back from internal Graph representation
        animate_graph_evolution(target_graph, optimiser.history, graph_gen_params.adapter, "./")

        optimiser.history.show.fitness_line()
    return found_graphs


def animate_graph_evolution(target_graph: nx.Graph, evolution_history: OptHistory, adapter: BaseOptimizationAdapter, dir_to_save_gif: str):
    last_internal_graph = evolution_history.archive_history[-1][0]

    # Choose nearest parent each time:
    genealogical_path: List[Individual] = [last_internal_graph]
    while genealogical_path[-1].parents:
        genealogical_path.append(max(
            genealogical_path[-1].parents,
            key=partial(adapter.adapt_func(tree_edit_dist), genealogical_path[-1])
        ))
        print(f"Generation: {genealogical_path[-1].native_generation}")

    print(genealogical_path)
    domain_evolutionary_path = list(reversed(adapter.restore(genealogical_path)))
    print(domain_evolutionary_path)

    target_frames = 10
    target_time_s = 3.

    # TODO: Make work for len(evolution_history) smaller than target frames, analyze typical situation
    # evolution_history = evolution_history[::len(evolution_history) // target_frames]

    fig, (target_ax, evo_ax) = plt.subplots(1, 2)

    def draw_graph(graph, ax, title):
        ax.clear()
        ax.set_title(title)
        colors, labeldict, legend_handles = _get_node_colors_and_labels(graph, False)
        nx.draw(graph, ax=ax, arrows=True, node_color=colors, with_labels=False, labels=labeldict)
        return legend_handles

    legend_handles = draw_graph(target_graph, target_ax, "Target graph")
    fig.legend(handles=legend_handles)

    def render_frame(frame_index):
        draw_graph(domain_evolutionary_path[frame_index], evo_ax, "Evolution process")
        return evo_ax,

    frames = len(domain_evolutionary_path)
    seconds_per_frame = target_time_s / frames
    fps = round(1 / seconds_per_frame)

    anim = animation.FuncAnimation(fig, render_frame, repeat=False, frames=frames, interval=1000*seconds_per_frame)

    anim.save(os.path.join(dir_to_save_gif, "evolution_process.gif"), fps=fps)
    plt.show()

if __name__ == '__main__':
    """
    In this example Optimizer is expected to find the target graph
    using Tree Edit Distance metric and a random tree (nx.random_tree) as target.
    The convergence can be seen from achieved metrics and visually from graph plots.
    """
    run_graph_search(visualize=True)
