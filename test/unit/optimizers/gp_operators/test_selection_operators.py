from golem.core.adapter import DirectAdapter
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.selection import Selection, SelectionTypesEnum, random_selection
from golem.core.optimisers.opt_history_objects.individual import Individual
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth, graph_fifth, RandomMetric


def get_population():
    adapter = DirectAdapter()
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    population = [Individual(adapter.adapt(graph)) for graph in graphs]
    for ind in population:
        ind.set_evaluation_result(SingleObjFitness(obj_function()))
    return population


def obj_function() -> float:
    metric_function = RandomMetric.get_value
    return metric_function()


def test_tournament_selection():
    num_of_inds = 3
    population = get_population()
    requirements = GPAlgorithmParameters(selection_types=[SelectionTypesEnum.tournament], pop_size=num_of_inds)
    selection = Selection(requirements)
    selected_individuals = selection(population)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_random_selection():
    num_of_inds = 2
    population = get_population()
    selected_individuals = random_selection(population, pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_random_individuals():
    num_of_inds = 2
    population = get_population()
    types = [SelectionTypesEnum.tournament]
    requirements = GPAlgorithmParameters(selection_types=types, pop_size=num_of_inds)
    selection = Selection(requirements)
    selected_individuals = selection.individuals_selection(individuals=population)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(set(selected_individuals_ref)) == len(selected_individuals) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_equality_individuals():
    num_of_inds = 4
    one_ind = get_population()[0]
    types = [SelectionTypesEnum.tournament]
    requirements = GPAlgorithmParameters(selection_types=types, pop_size=num_of_inds)
    population = [one_ind for _ in range(4)]
    selection = Selection(requirements)
    selected_individuals = selection.individuals_selection(individuals=population)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(selected_individuals) == num_of_inds and
            len(set(selected_individuals_ref)) == 1)
