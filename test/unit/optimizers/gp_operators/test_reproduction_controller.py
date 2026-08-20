import random
from math import ceil
from typing import Optional

import numpy as np
import pytest

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import Crossover, CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
from golem.core.optimisers.genetic.operators.selection import Selection, SelectionTypesEnum
from golem.core.optimisers.genetic.parameters.population_size import ConstRatePopulationSize
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError


def mock_mutation(graph, *args, **kwargs):
    return graph


class MockEvaluator(EvaluationOperator):
    def __init__(self, success_prob: float = 1.0):
        self.success_prob = success_prob

    def __call__(self, pop):
        n_valid = int(ceil(self.success_prob * len(pop)))
        evaluated = random.sample(pop, n_valid)
        return evaluated

    def eval_single(self, graph) -> Optional:
        is_valid = random.random() < self.success_prob
        return graph if is_valid else None


def get_rand_population(pop_size: int = 10) -> PopulationT:
    graph_sizes = list(range(5, 15))
    random_pop = [generate_labeled_graph('tree', size=random.choice(graph_sizes),
                                         directed=True)
                  for _ in range(pop_size)]
    graph_pop = BaseNetworkxAdapter().adapt(random_pop)
    individuals = [Individual(graph) for graph in graph_pop]
    return individuals


@pytest.fixture()
def reproducer() -> ReproductionController:
    # population size must grow each iteration
    params = GPAlgorithmParameters(pop_size=30, max_pop_size=100, offspring_rate=0.2,
                                   required_valid_ratio=0.9,
                                   mutation_types=[MutationTypesEnum.single_add,
                                                   MutationTypesEnum.single_drop],
                                   crossover_types=[CrossoverTypesEnum.none])
    graph_gen_params = GraphGenerationParams(available_node_types=['x'],
                                             rules_for_constraint=[])
    requirements = GraphRequirements()

    mutation = Mutation(params, requirements, graph_gen_params)
    crossover = Crossover(params, requirements, graph_gen_params)
    selection = Selection(params, requirements)

    reproduction = ReproductionController(params, selection, mutation, crossover)
    return reproduction


@pytest.mark.parametrize('success_rate', [0.4, 0.5, 0.9, 1.0])
def test_mean_success_rate(reproducer: ReproductionController, success_rate: float):
    """Tests that Reproducer correctly estimates average success rate"""
    assert np.isclose(reproducer.mean_success_rate, 1.0)

    evaluator = MockEvaluator(success_rate)
    pop = get_rand_population(reproducer.parameters.pop_size)
    num_iters = 50
    for i in range(num_iters):
        pop = reproducer.reproduce(pop, evaluator)

    assert np.isclose(reproducer.mean_success_rate, success_rate, rtol=0.1)


@pytest.mark.parametrize('success_rate', [0.0, 0.1])
def test_too_little_valid_evals(reproducer: ReproductionController, success_rate: float):
    evaluator = MockEvaluator(success_rate)
    pop = get_rand_population(reproducer.parameters.pop_size)

    with pytest.raises(EvaluationAttemptsError):
        reproducer.reproduce(pop, evaluator)


@pytest.mark.parametrize('success_rate', [0.2])
def test_minimal_valid_evals(reproducer: ReproductionController, success_rate: float):
    parameters = reproducer.parameters
    evaluator = MockEvaluator(success_rate)
    pop = get_rand_population(parameters.pop_size)
    num_iters = 10
    for i in range(num_iters):
        pop = reproducer.reproduce(pop, evaluator)
        actual_valid_ratio = len(pop) / parameters.pop_size
        assert parameters.required_valid_ratio > actual_valid_ratio >= reproducer._minimum_valid_ratio


@pytest.mark.parametrize('success_rate', [0.4, 0.9, 1.0])
def test_pop_size_progression(reproducer: ReproductionController, success_rate: float):
    parameters = reproducer.parameters
    required_valid = parameters.required_valid_ratio
    pop_size_progress = ConstRatePopulationSize(parameters.pop_size,
                                                parameters.offspring_rate,
                                                parameters.max_pop_size)

    evaluator = MockEvaluator(success_rate)
    pop = get_rand_population(parameters.pop_size)
    num_iters = 50
    for i in range(num_iters):
        prev_pop = pop
        pop = reproducer.reproduce(pop, evaluator)
        actual_pop_size = len(pop)

        # test that even with noisy evaluators we have steady increase in offsprings
        if i > 1:
            assert (actual_pop_size > len(prev_pop) or
                    actual_pop_size >= parameters.max_pop_size * required_valid)
        # and that this increase follows the one from parameters
        assert 1.0 >= (actual_pop_size / parameters.pop_size) >= required_valid

        # update pop size
        parameters.pop_size = pop_size_progress.next(pop)


def test_mating_selection_lifts_the_mating_pool_cap():
    """Without a mating selection the controller caps its request at the size
    of the population it selects from, and every selection operator answers
    such a request by returning everyone -- so reproduction applies no
    selection pressure at all. A mating selection that samples with
    replacement lifts the cap, and the request survives to the operator."""
    params = GPAlgorithmParameters(
        pop_size=20,
        mutation_types=[MutationTypesEnum.single_add],
        crossover_types=[CrossoverTypesEnum.none],
        mating_selection_types=[SelectionTypesEnum.tournament_with_replacement])
    graph_gen_params = GraphGenerationParams(available_node_types=['x'], rules_for_constraint=[])
    requirements = GraphRequirements()

    requested = []

    class RecordingSelection(Selection):
        def __call__(self, population, pop_size=None):
            requested.append(pop_size)
            return super().__call__(population, pop_size)

    mating = RecordingSelection(
        params, requirements,
        selection_types=[SelectionTypesEnum.tournament_with_replacement])
    reproduction = ReproductionController(
        params,
        selection=Selection(params, requirements),
        mutation=Mutation(params, requirements, graph_gen_params),
        crossover=Crossover(params, requirements, graph_gen_params),
        mating_selection=mating)

    population = get_rand_population(10)
    reproduction.reproduce(population, MockEvaluator(1.0))

    assert requested, 'the mating selection was never consulted'
    assert max(requested) > len(population)


def test_without_mating_selection_the_pool_never_exceeds_the_population():
    """Pins the historical behaviour the option above opts out of."""
    params = GPAlgorithmParameters(
        pop_size=20,
        mutation_types=[MutationTypesEnum.single_add],
        crossover_types=[CrossoverTypesEnum.none])
    graph_gen_params = GraphGenerationParams(available_node_types=['x'], rules_for_constraint=[])
    requirements = GraphRequirements()

    requested = []

    class RecordingSelection(Selection):
        def __call__(self, population, pop_size=None):
            requested.append(pop_size)
            return super().__call__(population, pop_size)

    reproduction = ReproductionController(
        params,
        selection=RecordingSelection(params, requirements),
        mutation=Mutation(params, requirements, graph_gen_params),
        crossover=Crossover(params, requirements, graph_gen_params))

    population = get_rand_population(10)
    reproduction.reproduce(population, MockEvaluator(1.0))

    assert requested
    assert max(requested) <= len(population)
