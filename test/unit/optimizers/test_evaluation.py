import datetime

import pytest

from golem.core.adapter import DirectAdapter
from golem.core.dag.graph import Graph
from golem.core.optimisers.fitness import Fitness, null_fitness
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher, SimpleDispatcher, \
    ObjectiveEvaluationDispatcher
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.timer import OptimisationTimer
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth, RandomMetric


def set_up_tests():
    adapter = DirectAdapter()
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth()]
    population = [Individual(adapter.adapt(graph)) for graph in graphs]
    return adapter, population


def get_objective(graph: Graph) -> Fitness:
    objective = Objective({'random_metric': RandomMetric.get_value})
    return objective(graph)


def invalid_objective(graph: Graph) -> Fitness:
    return null_fitness()


@pytest.mark.parametrize(
    'dispatcher',
    [SimpleDispatcher(DirectAdapter()),
     MultiprocessingDispatcher(DirectAdapter()),
     MultiprocessingDispatcher(DirectAdapter(), n_jobs=-1)]
)
def test_dispatchers_with_and_without_multiprocessing(dispatcher):
    _, population = set_up_tests()

    evaluator = dispatcher.dispatch(get_objective)
    evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(population) == len(evaluated_population), "Not all graphs was evaluated"


@pytest.mark.parametrize(
    'objective',
    [invalid_objective]
)
@pytest.mark.parametrize(
    'dispatcher',
    [MultiprocessingDispatcher(DirectAdapter()),
     SimpleDispatcher(DirectAdapter())]
)
def test_dispatchers_with_faulty_objectives(objective, dispatcher):
    adapter, population = set_up_tests()

    evaluator = dispatcher.dispatch(objective)
    assert evaluator(population) is None


@pytest.mark.parametrize('dispatcher', [
    MultiprocessingDispatcher(DirectAdapter()),
    SimpleDispatcher(DirectAdapter()),
])
def test_dispatcher_with_timeout(dispatcher: ObjectiveEvaluationDispatcher):
    adapter, population = set_up_tests()

    timeout = datetime.timedelta(seconds=0.00001)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = dispatcher.dispatch(get_objective, timer=t)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(evaluated_population) >= 1, "At least one graphs is evaluated"
    assert len(evaluated_population) < len(population), "Not all graphs should be evaluated (not enough time)"

    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = dispatcher.dispatch(get_objective, timer=t)
        evaluated_population = evaluator(population)
    fitness = [x.fitness for x in evaluated_population]
    assert all(x.valid for x in fitness), "At least one fitness value is invalid"
    assert len(population) == len(evaluated_population), "Not all graphs was evaluated"
