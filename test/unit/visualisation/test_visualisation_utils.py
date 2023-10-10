from golem.core.adapter import DirectAdapter
from golem.core.dag.convert import graph_structure_as_nx_graph
from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.visualisation.graph_viz import GraphVisualizer
from golem.visualisation.opt_viz_extra import extract_objectives
from test.unit.utils import graph_first


def make_comparable_lists(pos, real_hierarchy_levels, node_labels, dim, reverse):
    def extract_levels(hierarchy_levels):
        levels = []
        for pair in hierarchy_levels:
            levels.append(sorted(pair[1]))
        return levels

    computed_hierarchy_levels = {}
    for node in pos:
        level = pos[node][dim]
        if level in computed_hierarchy_levels:
            computed_hierarchy_levels[level].append(node_labels[node])
        else:
            computed_hierarchy_levels[level] = [node_labels[node]]

    sorted_computed_hierarchy_levels = sorted(computed_hierarchy_levels.items(),
                                              key=lambda x: x[0], reverse=reverse)
    sorted_real_hierarchy_levels = sorted(real_hierarchy_levels.items(),
                                          key=lambda x: x[0])
    return extract_levels(sorted_computed_hierarchy_levels), extract_levels(sorted_real_hierarchy_levels)


def test_hierarchy_pos():
    graph = graph_first()
    real_hierarchy_levels_y = {0: ['c'],
                               1: ['d', 'a'],
                               2: ['a'],
                               3: ['c', 'b'],
                               4: ['d']}
    real_hierarchy_levels_x = {0: ['c', 'd', 'c', 'd'],
                               1: ['a', 'b'],
                               2: ['a']}

    nx_graph, nodes_dict = graph_structure_as_nx_graph(graph)
    node_labels = {uid: str(node) for uid, node in nodes_dict.items()}

    pos, _, _ = GraphVisualizer._get_hierarchy_pos(nx_graph, nodes_dict)
    comparable_lists_y = make_comparable_lists(pos, real_hierarchy_levels_y,
                                               node_labels, 1, reverse=True)
    comparable_lists_x = make_comparable_lists(pos, real_hierarchy_levels_x,
                                               node_labels, 0, reverse=False)
    assert comparable_lists_y[0] == comparable_lists_y[1]  # check nodes hierarchy by y axis
    assert comparable_lists_x[0] == comparable_lists_x[1]  # check nodes hierarchy by x axis


def test_extract_objectives():
    num_of_inds = 5
    opt_graph = DirectAdapter().adapt(graph_first())
    individuals = [Individual(opt_graph) for _ in range(num_of_inds)]
    fitness = (-0.8, 0.1)
    weights = tuple([-1 for _ in range(len(fitness))])
    for ind in individuals:
        ind.set_evaluation_result(MultiObjFitness(values=fitness, weights=weights))
    populations_num = 3
    individuals_history = [individuals for _ in range(populations_num)]
    all_objectives = extract_objectives(individuals=individuals_history, transform_from_minimization=True)
    assert all_objectives[0][0] > 0 and all_objectives[0][2] > 0
    assert all_objectives[1][0] > 0 and all_objectives[1][2] > 0
