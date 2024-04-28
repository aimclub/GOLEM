from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.graph_search import graph_search_setup
from golem.utilities.utilities import set_random_seed


def test_random_seed_fully_determines_evolution_process():
    """ Tests that random seed fully determines evolution process. """
    # Setup graph search
    target_graph = generate_labeled_graph('tree', 4, node_labels=['X', 'Y'])

    def launch_with_seed(seed):
        set_random_seed(seed)
        optimizer, objective = graph_search_setup(
            target_graph=target_graph,
            num_iterations=3,
            node_types=['X', 'Y'],
            pop_size=3
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
