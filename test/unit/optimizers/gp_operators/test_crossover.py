from golem.core.adapter import DirectAdapter
from golem.core.optimisers.fitness import SingleObjFitness
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, Crossover, \
    exchange_edges_crossover, exchange_parents_one_crossover, exchange_parents_both_crossover, one_point_crossover
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from test.unit.utils import graph_first, graph_second, graph_sixth, graph_seventh, graph_eighth, graph_ninth, \
    graph_with_single_node, graph_with_multi_roots_first, graph_with_multi_roots_second
import pytest


@pytest.mark.parametrize('crossover_type', CrossoverTypesEnum)
def test_crossover_zero_probability(crossover_type):
    graph_example_first = graph_first()
    graph_example_second = graph_second()
    
    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd'])
    parameters = GPAlgorithmParameters(crossover_prob=0)
    crossover = Crossover(parameters, requirements, graph_generation_params)

    crossover.parameters.crossover_types = [crossover_type]
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second    


def test_crossover_none():
    graph_example_first = graph_first()
    graph_example_second = graph_second()

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd'])
    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.none], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second


def test_crossover_exchange_edges():
    graph_example_first = graph_sixth()
    graph_example_second = graph_seventh()
    valid_graphs = [graph_example_first, graph_example_second, graph_eighth(), graph_ninth()]

    new_graphs = exchange_edges_crossover(graph_example_first, graph_example_second, 2)
    assert any([new_graphs[0] == graph for graph in valid_graphs])
    assert any([new_graphs[1] == graph for graph in valid_graphs])    


def test_crossover_exchange_parents_one():
    graph_example_first = graph_sixth()
    graph_example_second = graph_seventh()
    valid_graphs = [graph_example_first, graph_example_second]

    new_graphs = exchange_parents_one_crossover(graph_example_first, graph_example_second, 2)
    assert any([new_graphs[0] == graph for graph in valid_graphs])
    assert any([new_graphs[1] == graph for graph in valid_graphs])  


def test_crossover_exchange_parents_both():
    graph_example_first = graph_sixth()
    graph_example_second = graph_seventh()
    valid_graphs = [graph_example_first, graph_example_second]

    new_graphs = exchange_parents_both_crossover(graph_example_first, graph_example_second, 2)
    assert any([new_graphs[0] == graph for graph in valid_graphs])
    assert any([new_graphs[1] == graph for graph in valid_graphs])  


def test_one_point_sink_filter_respected_on_multi_root_graphs():
    """The sink guard must work for graphs with several roots as well: a pair that
    would place a rejected node into a root position may not be swapped."""
    reference_roots = (set(n.descriptive_id for n in graph_with_multi_roots_first().root_nodes()),
                       set(n.descriptive_id for n in graph_with_multi_roots_second().root_nodes()))
    for _ in range(30):
        first, second = graph_with_multi_roots_first(), graph_with_multi_roots_second()
        one_point_crossover(first, second, max_depth=5, sink_filter=lambda node: False)
        result_roots = (set(n.descriptive_id for n in first.root_nodes()),
                        set(n.descriptive_id for n in second.root_nodes()))
        assert result_roots == reference_roots, \
            'a reject-all sink filter must keep every root set unchanged'


def test_crossover_returns_evaluated_parents_when_sink_filter_rejects_all():
    """When no sink-valid crossover pick exists, the original (already evaluated)
    individuals must be returned instead of unchanged copies with null fitness,
    which would only waste a duplicate evaluation."""
    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd'])
    graph_generation_params.advisor.can_be_sink = lambda node: False
    parameters = GPAlgorithmParameters(crossover_prob=1,
                                       crossover_types=[CrossoverTypesEnum.subtree,
                                                        CrossoverTypesEnum.one_point])
    crossover = Crossover(parameters, requirements, graph_generation_params)

    for _ in range(20):
        ind_first = Individual(DirectAdapter().adapt(graph_first()))
        ind_second = Individual(DirectAdapter().adapt(graph_second()))
        ind_first.set_evaluation_result(SingleObjFitness(1))
        ind_second.set_evaluation_result(SingleObjFitness(2))
        new_first, new_second = crossover._crossover(ind_first, ind_second)
        noop_with_null_fitness = (new_first.graph == ind_first.graph and
                                  new_second.graph == ind_second.graph and
                                  not (new_first.fitness.valid and new_second.fitness.valid))
        assert not noop_with_null_fitness, \
            'offspring identical to the parents must reuse their evaluated fitness'


@pytest.mark.parametrize('crossover_type', CrossoverTypesEnum)
def test_crossover_with_single_node(crossover_type):
    graph_example_first = graph_with_single_node()
    graph_example_second = graph_with_single_node()

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd'])
    parameters = GPAlgorithmParameters(crossover_prob=1)
    crossover = Crossover(parameters, requirements, graph_generation_params)

    crossover.parameters.crossover_types = [crossover_type]
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second    