import logging
from functools import partial

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.api.main import GOLEM
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.metrics.edit_distance import tree_edit_dist


def run_graph_search(size=16, timeout=8, visualize=True):
    # Generate target graph that will be sought by optimizer
    node_types = ('a', 'b')
    target_graph = generate_labeled_graph('tree', size, node_labels=node_types)

    # Generate initial population with small tree graphs
    initial_graphs = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]
    # Setup objective: edit distance to target graph
    objective = Objective(partial(tree_edit_dist, target_graph))

    golem = GOLEM(timeout=timeout,
                  logging_level=logging.INFO,
                  early_stopping_iterations=100,
                  initial_graphs=initial_graphs,
                  objective=objective,
                  genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
                  max_pop_size=50,
                  mutation_types=[MutationTypesEnum.single_add,
                                  MutationTypesEnum.single_drop,
                                  MutationTypesEnum.single_change],
                  crossover_types=[CrossoverTypesEnum.subtree],
                  available_node_types=node_types  # Node types that can appear in graphs
                  )
    found_graphs = golem.optimise()

    return found_graphs


if __name__ == '__main__':
    """
    Same as `simple_run.py` but with GOLEM API usage example.
    In this example Optimizer is expected to find the target graph
    using Tree Edit Distance metric and a random tree (nx.random_tree) as target.
    The convergence can be seen from achieved metrics and visually from graph plots.
    """
    run_graph_search(visualize=True)
