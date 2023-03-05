from golem.core.adapter import DirectAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, Crossover
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from test.unit.utils import graph_first, graph_second, graph_sixth, graph_seventh, graph_eighth, graph_ninth, graph_with_single_node


def test_crossover():
    adapter = DirectAdapter()
    graph_example_first = adapter.adapt(graph_first())
    graph_example_second = adapter.adapt(graph_second())

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd', 'e', 'f'])
    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.none], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second

    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.subtree], crossover_prob=0)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second

    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.exchange_edges], crossover_prob=0)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second    


    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.exchange_parents_one], crossover_prob=0)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second       


    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.exchange_parents_both], crossover_prob=0)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second       


def test_crossover_exchange_edges():
    adapter = DirectAdapter()
    graph_example_first = adapter.adapt(graph_sixth())
    graph_example_second = adapter.adapt(graph_seventh())
    valid_graphs = [graph_example_first, graph_example_first, adapter.adapt(graph_eighth()), adapter.adapt(graph_ninth())]

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c'])
    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.exchange_edges], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert ([new_graphs[0].graph == graph for graph in valid_graphs] != [])
    assert ([new_graphs[1].graph == graph for graph in valid_graphs] != [])    


def test_crossover_exchange_parents_one():
    adapter = DirectAdapter()
    graph_example_first = adapter.adapt(graph_sixth())
    graph_example_second = adapter.adapt(graph_seventh())
    valid_graphs = [graph_example_first, graph_example_first]

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c'])
    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.exchange_parents_one], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert ([new_graphs[0].graph == graph for graph in valid_graphs] != [])
    assert ([new_graphs[1].graph == graph for graph in valid_graphs] != [])


def test_crossover_exchange_parents_both():
    adapter = DirectAdapter()
    graph_example_first = adapter.adapt(graph_sixth())
    graph_example_second = adapter.adapt(graph_seventh())
    valid_graphs = [graph_example_first, graph_example_first]

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c'])
    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.exchange_parents_both], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, graph_generation_params)
    new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])
    assert ([new_graphs[0].graph == graph for graph in valid_graphs] != [])
    assert ([new_graphs[1].graph == graph for graph in valid_graphs] != [])


def test_crossover_with_single_node():
    adapter = DirectAdapter()
    graph_example_first = adapter.adapt(graph_with_single_node())
    graph_example_second = adapter.adapt(graph_with_single_node())

    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd', 'e', 'f'])

    for crossover_type in CrossoverTypesEnum:
        opt_parameters = GPAlgorithmParameters(crossover_types=[crossover_type], crossover_prob=1)
        crossover = Crossover(opt_parameters, requirements, graph_generation_params)
        new_graphs = crossover([Individual(graph_example_first), Individual(graph_example_second)])

        assert new_graphs[0].graph == graph_example_first
        assert new_graphs[1].graph == graph_example_second
