from copy import deepcopy

from golem.core.adapter import DirectAdapter
from golem.core.dag.graph_utils import nodes_from_layer
from golem.core.optimisers.archive import ParetoFront
from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.genetic.gp_operators import filter_duplicates, replace_subtrees
from golem.core.optimisers.opt_history_objects.individual import Individual
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth


def test_filter_duplicates():
    archive = ParetoFront()
    adapter = DirectAdapter()

    archive_items = [Individual(adapter.adapt(g)) for g in [graph_first(), graph_second(), graph_third()]]
    population = [Individual(adapter.adapt(g)) for g in [graph_first(), graph_second(),
                                                         graph_third(), graph_fourth()]]
    archive_items_fitness = ((0.80001, 0.25), (0.7, 0.1), (0.9, 0.7))
    population_fitness = ((0.8, 0.25), (0.59, 0.25), (0.9, 0.7), (0.7, 0.1))
    weights = (-1, 1)
    for ind_num in range(len(archive_items)):
        archive_items[ind_num].set_evaluation_result(
            MultiObjFitness(values=archive_items_fitness[ind_num], weights=weights))
    for ind_num in range(len(population)):
        population[ind_num].set_evaluation_result(MultiObjFitness(values=population_fitness[ind_num], weights=weights))
    archive.update(archive_items)
    filtered_archive = filter_duplicates(archive, population)
    assert len(filtered_archive) == 1
    assert filtered_archive[0].fitness.values[0] == -0.80001
    assert filtered_archive[0].fitness.values[1] == 0.25


def test_replace_subtree():
    # graph with depth = 3
    graph_1 = graph_first()
    passed_graph_1 = deepcopy(graph_1)
    # graph with depth = 2
    graph_2 = graph_third()

    # choose the first layer of the first graph
    layer_in_first = graph_1.depth - 1
    # choose the last layer of the second graph
    layer_in_second = 0
    max_depth = 3

    node_from_graph_first = nodes_from_layer(graph_1, layer_in_first)[0]
    node_from_graph_second = nodes_from_layer(graph_2, layer_in_second)[0]

    # replace_subtrees must not replace subgraph in the first graph and its depth must be <= max_depth
    replace_subtrees(graph_1, graph_2, node_from_graph_first, node_from_graph_second,
                     layer_in_first, layer_in_second, max_depth)
    assert graph_1.depth <= max_depth
    assert graph_1 == passed_graph_1
    assert graph_2.depth <= max_depth
