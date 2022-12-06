from functools import reduce
from random import choice

import pytest

from golem.core.adapter import DirectAdapter
from golem.core.optimisers.initial_graphs_generator import InitialPopulationGenerator
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.test.unit.utils import graph_first, graph_second, graph_third


def setup_test(pop_size):
    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd', 'e', 'f'])
    generator = InitialPopulationGenerator(pop_size, graph_generation_params, requirements)
    return requirements, graph_generation_params, generator


def test_initial_graphs_as_initial_population():
    adapter = DirectAdapter()
    initial_graphs = adapter.adapt([graph_first(), graph_second(), graph_third()])

    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=3)
    initial_population_generator.with_initial_graphs(initial_graphs)
    generated_population = initial_population_generator()
    assert generated_population == initial_graphs

    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=4)
    initial_population_generator.with_initial_graphs(initial_graphs)
    generated_population = initial_population_generator()
    assert generated_population == initial_graphs

    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=2)
    initial_population_generator.with_initial_graphs(initial_graphs)
    generated_population = initial_population_generator()
    assert len(generated_population) == 2
    assert all(graph in initial_graphs for graph in generated_population)


@pytest.mark.parametrize('pop_size', [3, 4])
def test_initial_population_generation_function(pop_size):
    requirements, graph_generation_params, initial_population_generator = setup_test(pop_size=pop_size)
    initial_population_generator.with_custom_generation_function(
        lambda: choice([graph_first(), graph_second(), graph_third()]))
    verifier = graph_generation_params.verifier

    generated_population = initial_population_generator()
    assert len(generated_population) <= 3
    assert all(verifier(graph) for graph in generated_population)
    unique = reduce(lambda l, x: l.append(x) or l if x not in l else l, generated_population, [])
    assert len(unique) == len(generated_population)
