from datetime import timedelta

from examples.synthetic_graph_evolution.experiment_setup import run_trial
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from examples.synthetic_graph_evolution.tree_search import tree_search_setup


DIVERSITY_THRESHOLD = 0.6


def test_structural_diversity():
    """ Checks population's structural diversity. Diversity should not be lower than DIVERSITY_THRESHOLD. """
    target_graph = generate_labeled_graph('tree', 4, node_labels=['x'])
    node_types = ['x', 'y', 'z', 'w', 'v', 'u']
    # running the example
    _, history = run_trial(target_graph=target_graph,
                           optimizer_setup=tree_search_setup,
                           node_types=node_types,
                           timeout=timedelta(minutes=1))

    h = history.generations[:-1]
    ratio_unique = [len(set(ind.graph.descriptive_id for ind in pop)) / len(pop) for pop in h]
    for i in range(len(ratio_unique)):
        if i % 5 == 0:
            # structural check is applied every 5-th generation
            assert ratio_unique[i] == 1
        else:
            assert ratio_unique[i] > DIVERSITY_THRESHOLD
