from itertools import zip_longest
from math import ceil
import random
from typing import Optional

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator, PopulationT
from golem.core.optimisers.genetic.operators.regularization import Regularization, RegularizationTypesEnum
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimizer import GraphGenerationParams


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


def get_regularization_instance(regularization_type: RegularizationTypesEnum) -> Regularization:
    params = GPAlgorithmParameters(pop_size=30, max_pop_size=100, offspring_rate=0.2,
                                   required_valid_ratio=0.9,
                                   regularization_type=regularization_type)
    graph_gen_params = GraphGenerationParams(available_node_types=['x'],
                                             rules_for_constraint=[])
    regularization = Regularization(parameters=params, graph_generation_params=graph_gen_params)
    return regularization


def test_non_regularization():
    population = get_rand_population()
    regularization = get_regularization_instance(RegularizationTypesEnum.none)
    regularized_population = regularization(population=population, evaluator=MockEvaluator())
    assert all(x == y for x, y in zip_longest(population, regularized_population))


# def test_decremental_regularization():
#     population = get_rand_population()
#     regularization = get_regularization_instance(RegularizationTypesEnum.decremental)
#     regularized_population = regularization(population=population, evaluator=MockEvaluator())
#     # TODO: add asserts
