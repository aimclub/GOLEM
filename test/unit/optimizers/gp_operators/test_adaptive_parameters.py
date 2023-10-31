import random
from itertools import product
from typing import Any, Tuple, List

import pytest

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.archive import GenerationKeeper
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.genetic.parameters.graph_depth import AdaptiveGraphDepth
from golem.core.optimisers.genetic.parameters.mutation_prob import AdaptiveMutationProb
from golem.core.optimisers.genetic.parameters.operators_prob import AdaptiveVariationProb
from golem.core.optimisers.genetic.parameters.parameter import AdaptiveParameter, ConstParameter
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual


def get_rand_population(pop_size: int = 10, fitness_edges: Tuple[float, float] = (0, 1)) -> PopulationT:
    graph_sizes = list(range(5, 15))
    random_pop = [generate_labeled_graph('tree', size=random.choice(graph_sizes),
                                         directed=True)
                  for _ in range(pop_size)]
    graph_pop = BaseNetworkxAdapter().adapt(random_pop)
    individuals = [Individual(graph, fitness=SingleObjFitness(random.uniform(*fitness_edges))) for graph in graph_pop]
    return individuals


def get_adaptive_depth(start_depth: int, max_depth: int, max_stagnation_gens: int = 1, adaptive: bool = True):
    def custom_objective():
        return
    objective = Objective({'custom': custom_objective})
    generation_keeper = GenerationKeeper(objective=objective)
    adaptive_depth = AdaptiveGraphDepth(generation_keeper, start_depth=start_depth,
                                        max_depth=max_depth, max_stagnation_gens=max_stagnation_gens,
                                        adaptive=adaptive)
    return adaptive_depth, generation_keeper

def _test_adaptive_probability_parameter(adaptive_parameter: AdaptiveParameter, check_uniqueness: bool = True):
    def return_list(data: Any):
        return list(data) if hasattr(data, '__len__') else [data]

    probs = return_list(adaptive_parameter.initial)
    for _ in range(10):
        probs.extend(return_list(adaptive_parameter.next(get_rand_population())))

    if check_uniqueness:
        assert len(set(probs)) > 1

    for prob in probs:
        assert 0 < prob <= 1


@pytest.mark.parametrize('default_prob', (-100, -1, -1e-12, 1 + 1e-12, 100, None))
def test_adaptive_mutation_prob_with_uncorrect_params(default_prob: float):
    with pytest.raises(ValueError):
        AdaptiveMutationProb(default_prob)


@pytest.mark.parametrize('default_prob', (-100, -1, -1e-12, 1 + 1e-12, 100, None))
def test_adaptive_variation_prob_with_uncorrect_params(default_prob: float):
    with pytest.raises(ValueError):
        AdaptiveVariationProb(default_prob, default_prob)


@pytest.mark.parametrize('default_prob', [0, 0.5, 1])
def test_constant_parameter(default_prob: float):
    adaptive_parameter = ConstParameter(default_prob)
    assert default_prob == adaptive_parameter.initial
    for _ in range(10):
        assert default_prob == adaptive_parameter.next(get_rand_population())


@pytest.mark.parametrize('default_prob', [0.1, 0.5, 1])
def test_adaptive_mutation(default_prob: float):
    adaptive_parameter = AdaptiveMutationProb(default_prob=default_prob)
    _test_adaptive_probability_parameter(adaptive_parameter)


@pytest.mark.parametrize('default_prob', [0.1, 0.5, 1])
def test_operator_prob(default_prob: float):
    adaptive_parameter = AdaptiveVariationProb(mutation_prob=default_prob, crossover_prob=default_prob)
    _test_adaptive_probability_parameter(adaptive_parameter)


@pytest.mark.parametrize(['start_depth', 'max_depth', 'adaptive'],
                         [(1, 10, True), (2, 2, True), (1, 10, False)])
def test_adaptive_depth(start_depth: int, max_depth: int, adaptive: bool):
    adaptive_depth, generation_keeper = get_adaptive_depth(start_depth=start_depth,
                                                           max_depth=max_depth, adaptive=adaptive)
    assert adaptive_depth.initial == start_depth

    def step_with_new_population_with_new_fitness(delta: int, fitness_edges: List[int] = [0, 1]):
        """ create new population with fitness between fitness_edges[0] and fitness_edges[1]
            then add it to keeper and calculate new depth
            delta is step in fitness_edges difference """
        population = get_rand_population(fitness_edges=tuple(fitness_edges))
        fitness_edges[0] += delta
        fitness_edges[1] += delta
        generation_keeper.append(population)
        return adaptive_depth.next(population)

    # two tests:
    #     1. delta = 2, fitness grows, depth should grow if adaptive == True
    #     2. delta = -1, fitness reduces, depth should reduce if adaptive == True
    for delta in (2, -1):
        old_depth = step_with_new_population_with_new_fitness(delta)
        for i in range(10):
            depth = step_with_new_population_with_new_fitness(delta)
            assert depth <= max_depth
            if adaptive and delta > 0:
                assert old_depth < depth or depth == max_depth
            elif adaptive and delta < 0:
                assert old_depth >= depth or depth == max_depth
            else:
                assert depth in (start_depth, max_depth)
            old_depth = depth


@pytest.mark.parametrize(['start_depth', 'max_depth', 'max_stagnation_gens'],
                         [(-1, 10, 1),
                          (0, 10, 1),
                          (10, 1, 1),
                          (1, -2, 1),
                          (1, 10, -1),
                          (1, 10, 0),
                          (None, 10, 1),
                          (1, None, 1),
                          (1, 10, None),
                          ])
def test_adaptive_depth_with_uncorrect_params(start_depth: int, max_depth: int, max_stagnation_gens: int):
    with pytest.raises(ValueError):
        get_adaptive_depth(start_depth=start_depth, max_depth=max_depth, max_stagnation_gens=max_stagnation_gens)
