import datetime
import logging
from functools import partial

import pickle

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.api.main import GOLEM
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.metrics.edit_distance import tree_edit_dist


def test_specifying_parameters_through_api():
    """ Tests that parameters for optimizer are specified correctly. """

    timeout = 1
    size = 16
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

    # setup with externally specifying params
    requirements = GraphRequirements(
        early_stopping_iterations=100,
        timeout=datetime.timedelta(minutes=timeout),
        n_jobs=1,
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

    assert golem.gp_algorithm_parameters == gp_params
    # compared by pickle dump since there are lots of inner classes with not implemented __eq__ magic methods
    # probably needs to be fixed
    assert pickle.dumps(golem.graph_generation_parameters) == pickle.dumps(graph_gen_params)
    # need to be compared by dicts since the classes itself are different
    assert golem.graph_requirements.__dict__ == requirements.__dict__
