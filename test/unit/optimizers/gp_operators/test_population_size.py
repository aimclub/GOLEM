import pytest

from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.genetic.parameters.population_size import PopulationSize, AdaptivePopulationSize, \
    ConstRatePopulationSize
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.utilities.sequence_iterator import SequenceIterator, fibonacci_sequence


def custom_objective():
    return


def pop_size_sequence(n: int) -> int:
    a = 3
    b = 8
    for __ in range(n):
        a, b = b, a + b
    return max(a, 3)


def test_const_pop_size_increases():
    """ If there are too many fitness evaluation errors pop_size must increase to save population.
     With const pop_size population size mustn't be lower than initial pop_size. """
    initial_pop_size = 20
    pop_size = ConstRatePopulationSize(pop_size=initial_pop_size,
                                       offspring_rate=1)

    # only one successfully evaluated individual
    population = [Individual(OptGraph(OptNode('rf')))]
    assert pop_size.next(population) >= initial_pop_size


def test_adaptive_pop_size_increases():
    """ If there are too many fitness evaluation errors pop_size must increase to save population.
     With adaptive pop_size population size must increase using iterator method `next`. """
    objective = Objective({'custom': custom_objective})
    generation_keeper = GenerationKeeper(objective=objective)
    pop_size = AdaptivePopulationSize(improvement_watcher=generation_keeper,
                                      progression_iterator=SequenceIterator(sequence_func=pop_size_sequence))

    # only one successfully evaluated individual
    base_graph = OptGraph(OptNode('rf'))
    # to test only `too_many_fitness_eval_errors` case without `no_progress`
    fitness = [SingleObjFitness(primary_value=-0.8),
               SingleObjFitness(primary_value=-1)]
    population_0 = [Individual(base_graph, fitness=fitness[0])]
    generation_keeper.append(population_0)
    population_1 = [Individual(base_graph, fitness=fitness[1])]
    assert pop_size.next(population_1) >= len(population_1)
