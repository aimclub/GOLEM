import logging

from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.genetic.operators.selection import (
    Selection, SelectionTypesEnum, random_selection, nsga2_selection,
    tournament_selection_with_replacement, fast_non_dominated_sort)
from golem.core.optimisers.opt_history_objects.individual import Individual
from test.unit.optimizers.test_evaluation import get_objective
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth, graph_fifth
from random import sample


def custom_selection(population: PopulationT, pop_size: int):
    return sample(population, pop_size)


def get_population():
    adapter = DirectAdapter()
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    population = [Individual(adapter.adapt(graph)) for graph in graphs]
    for ind in population:
        ind.set_evaluation_result(get_objective(ind.graph))
    return population


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
    selected_individuals = selection(population)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(set(selected_individuals_ref)) == len(selected_individuals) and
            len(selected_individuals) == num_of_inds)


def test_selection_with_explicit_zero_pop_size():
    """An explicit request for zero individuals must not silently fall back
    to the default population size."""
    population = get_population()
    requirements = GPAlgorithmParameters(selection_types=[SelectionTypesEnum.tournament], pop_size=4)
    selection = Selection(requirements)
    assert selection(population, pop_size=0) == []


def test_individuals_selection_equality_individuals():
    num_of_inds = 4
    one_ind = get_population()[0]
    types = [SelectionTypesEnum.tournament]
    requirements = GPAlgorithmParameters(selection_types=types, pop_size=num_of_inds)
    population = [one_ind for _ in range(4)]
    selection = Selection(requirements)
    selected_individuals = selection(population)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(selected_individuals) == num_of_inds and
            len(set(selected_individuals_ref)) == 1)


def test_custom_selection():
    num_of_inds = 3
    population = get_population()
    requirements = GPAlgorithmParameters(selection_types=[custom_selection], pop_size=num_of_inds)
    selection = Selection(requirements)
    selected_individuals = selection(population)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def _mo_population(objective_values):
    """Individuals carrying the given multi-objective fitness vectors.

    The graphs are irrelevant to these operators, which read fitness only.
    """
    adapter = DirectAdapter()
    population = []
    for values in objective_values:
        ind = Individual(adapter.adapt(graph_first()))
        ind.set_evaluation_result(MultiObjFitness(values=values))
        population.append(ind)
    return population


def test_nsga2_keeps_the_whole_non_dominated_front():
    # Three mutually non-dominated points, three dominated by all of them.
    front = [(1.0, 5.0), (3.0, 3.0), (5.0, 1.0)]
    dominated = [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)]
    population = _mo_population(front + dominated)

    selected = nsga2_selection(population, pop_size=3)

    assert len(selected) == 3
    assert {tuple(ind.fitness.values) for ind in selected} == set(front)


def test_nsga2_fills_from_the_next_front_when_the_first_is_too_small():
    front = [(1.0, 5.0), (5.0, 1.0)]
    dominated = [(6.0, 6.0), (7.0, 7.0)]
    population = _mo_population(front + dominated)

    selected = nsga2_selection(population, pop_size=3)

    assert len(selected) == 3
    assert set(front).issubset({tuple(ind.fitness.values) for ind in selected})


def test_nsga2_keeps_the_extremes_when_a_front_overflows():
    """Crowding distance is infinite at the ends of a front, so the extreme
    trade-offs survive truncation and the crowded middle is dropped."""
    population = _mo_population([(1.0, 9.0), (4.0, 4.1), (4.1, 4.0), (9.0, 1.0)])

    selected = nsga2_selection(population, pop_size=2)

    assert {tuple(ind.fitness.values) for ind in selected} == {(1.0, 9.0), (9.0, 1.0)}


def test_nsga2_is_reachable_through_the_selection_operator():
    population = _mo_population([(1.0, 5.0), (3.0, 3.0), (5.0, 1.0), (9.0, 9.0)])
    parameters = GPAlgorithmParameters(selection_types=[SelectionTypesEnum.nsga2],
                                       multi_objective=True, pop_size=3)
    selected = Selection(parameters)(population)
    assert len(selected) == 3


def test_fast_non_dominated_sort_orders_the_fronts():
    population = _mo_population([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    fronts = fast_non_dominated_sort(population)
    assert [[tuple(ind.fitness.values) for ind in front] for front in fronts] == \
           [[(1.0, 1.0)], [(2.0, 2.0)], [(3.0, 3.0)]]


def test_mating_pool_can_be_larger_than_the_population():
    """The whole point of sampling with replacement: reproduction asks for a
    pool that is not smaller than the population it selects from, and every
    other selection answers that by returning everyone."""
    population = get_population()
    pool = tournament_selection_with_replacement(population, pop_size=len(population) * 3)
    assert len(pool) == len(population) * 3
    assert all(ind in population for ind in pool)


def test_mating_pool_applies_selection_pressure():
    """With two candidates and binary tournaments, the dominating one wins
    every draw -- so a pool built with replacement contains only it, where a
    plain selection would have returned both untouched."""
    population = _mo_population([(1.0, 1.0), (9.0, 9.0)])
    best = tuple(population[0].fitness.values)

    pool = tournament_selection_with_replacement(population, pop_size=20)

    assert len(pool) == 20
    assert all(tuple(ind.fitness.values) == best for ind in pool)


def test_mating_pool_survives_a_degenerate_population():
    assert tournament_selection_with_replacement([], pop_size=5) == []
    single = get_population()[:1]
    assert len(tournament_selection_with_replacement(single, pop_size=5)) == 5


def test_multi_objective_generational_scheme_is_flagged(caplog):
    """That combination leaves the loop with no survival selection at all and
    fails silently, so it must at least say so."""
    with caplog.at_level(logging.WARNING):
        GPAlgorithmParameters(multi_objective=True,
                              genetic_scheme_type=GeneticSchemeTypesEnum.generational)
    assert any('no survival selection' in record.message for record in caplog.records)


def test_steady_state_multi_objective_is_not_flagged(caplog):
    with caplog.at_level(logging.WARNING):
        GPAlgorithmParameters(multi_objective=True,
                              genetic_scheme_type=GeneticSchemeTypesEnum.steady_state)
    assert not any('no survival selection' in record.message for record in caplog.records)
