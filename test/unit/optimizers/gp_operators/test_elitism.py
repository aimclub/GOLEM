import pytest

from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.elitism import Elitism, ElitismTypesEnum
from golem.core.optimisers.opt_history_objects.individual import Individual
from test.unit.optimizers.gp_operators.test_selection import get_objective
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth, graph_fifth


@pytest.fixture()
def set_up():
    adapter = DirectAdapter()
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    population = [Individual(adapter.adapt(graph)) for graph in graphs]
    for ind in population:
        ind.set_evaluation_result(get_objective(ind.graph))
    population, best_individuals = population[:4], population[2:]

    return best_individuals, population


def test_keep_n_best_elitism(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.keep_n_best))
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        # checks that new population contains the best individuals and `keep_n_best_elitism` does not duplicate it
        assert new_population.count(best_ind) == 1
    assert len(population) == len(new_population)


def test_keep_n_best_elitism_with_oversized_archive():
    """Elites must not fill the whole next generation: an archive of size >= pop_size
    would otherwise return the same individuals forever, freezing evolution."""
    adapter = DirectAdapter()
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]

    def make_population(n):
        # DirectAdapter deep-copies on adapt, so reusing source graphs is safe
        population = [Individual(adapter.adapt(graphs[i % len(graphs)])) for i in range(n)]
        for ind in population:
            ind.set_evaluation_result(get_objective(ind.graph))
        return population

    new_population = make_population(4)
    elitism = Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.keep_n_best))
    for archive_size in (4, 6):
        archive = make_population(archive_size)
        final_population = elitism.keep_n_best_elitism(archive, new_population)
        assert len(final_population) <= len(new_population)
        assert any(ind in new_population for ind in final_population), \
            'at least one new individual must survive elitism'


def test_replace_worst(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst))
    new_population = elitism(best_individuals, population)
    for ind in population:
        if ind not in new_population:
            assert all(ind.fitness <= best_ind.fitness for best_ind in new_population)
    assert len(new_population) == len(population)


def test_elitism_not_applicable(set_up):
    best_individuals, population = set_up
    elitisms = [
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                      multi_objective=True)),
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                      pop_size=4, min_pop_size_with_elitism=5)),
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.none)),
    ]
    for elitism in elitisms:
        new_population = elitism(best_individuals, population)
        assert new_population == population
