from typing import Sequence

from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.fitness import Fitness, MultiObjFitness, null_fitness
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from test.unit.utils import RandomMetric, DepthMetric


def create_individual(fitness: Fitness = None) -> Individual:
    first = OptNode(content={'name': 'n1'})
    graph = OptGraph(first)
    individual = Individual(graph)
    individual.set_evaluation_result(fitness or null_fitness())
    return individual


def create_population(fitness: Sequence[Fitness]) -> PopulationT:
    return tuple(map(create_individual, fitness))


def generation_keeper(init_population=None, multi_objective=True):
    quality_metrics = {'random_metric': RandomMetric.get_value}
    complexity_metrics = {'depth': DepthMetric.get_value}
    objective = Objective(quality_metrics=quality_metrics, complexity_metrics=complexity_metrics,
                          is_multi_objective=multi_objective)
    return GenerationKeeper(objective, initial_generation=init_population)


def population1():
    return create_population([
        MultiObjFitness([2, 4], weights=-1),
        MultiObjFitness([3, 2], weights=-1),
    ])


def population2():
    return create_population([
        MultiObjFitness([1, 5], weights=-1),
        MultiObjFitness([3, 3], weights=-1),
    ])


def test_archive_no_improvement():
    archive = generation_keeper(population1())
    assert archive.stagnation_iter_count == 0
    assert archive.is_any_improved
    assert archive.is_quality_improved and archive.is_complexity_improved
    assert archive.generation_num == 1

    archive.append(population1())
    assert archive.stagnation_iter_count == 1
    assert not archive.is_any_improved
    assert not archive.is_quality_improved and not archive.is_complexity_improved
    assert archive.generation_num == 2


def test_archive_multiobj_one_improvement():
    archive = generation_keeper(population1())
    previous_size = len(archive.best_individuals)

    # second population has dominating individuals
    assert any(new_ind.fitness.dominates(population1()[1].fitness)
               for new_ind in population2())
    archive.append(population2())

    assert archive.stagnation_iter_count == 0
    assert archive.is_any_improved
    assert archive.generation_num == 2
    # plus one non-dominated individual
    # minus one strongly dominated individual (substituted by better one)
    assert len(archive.best_individuals) == previous_size + 1
    assert archive.is_complexity_improved
    assert not archive.is_quality_improved
