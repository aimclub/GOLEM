from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from golem.utilities.utilities import set_random_seed
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum


def test_random_seed_fully_determines_evolution_process():
    """ Tests that random seed fully determines evolution process. """
    # Setup graph search
    target_graph = generate_labeled_graph('tree', 4, node_labels=['X', 'Y'])

    def launch_with_seed(seed):
        set_random_seed(seed)

        algorithm_parameters = GPAlgorithmParameters(
            adaptive_mutation_type=MutationAgentTypeEnum.random,
            pop_size=3,
            multi_objective=True,
            mutation_types=[
                MutationTypesEnum.single_add,
                MutationTypesEnum.single_edge,
                MutationTypesEnum.single_drop
            ],
            crossover_types=[
                CrossoverTypesEnum.one_point,
                CrossoverTypesEnum.subtree
            ],
        )

        optimizer, objective = graph_search_setup(
            target_graph=target_graph,
            num_iterations=10,
            node_types=['X', 'Y'],
            algorithm_parameters=algorithm_parameters
        )
        optimizer.optimise(objective)
        return optimizer.history

    def history_equals(history1, history2):
        def get_generation_ids(history):
            return [[ind.graph.descriptive_id for ind in generation] for generation in history.generations]
        return get_generation_ids(history1) == get_generation_ids(history2)

    seed = 42
    first_run = launch_with_seed(seed)
    second_run = launch_with_seed(seed)

    assert history_equals(first_run, second_run)
