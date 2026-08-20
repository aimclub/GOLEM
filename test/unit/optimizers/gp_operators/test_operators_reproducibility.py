from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, Crossover
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.utilities.utilities import set_random_seed
from test.unit.utils import graph_first, graph_second
from test.unit.optimizers.gp_operators.test_mutation import get_mutation_params
import pytest

# subgraph crossover is non-reproducible by design
crossover_types = [
    CrossoverTypesEnum.subtree,
    CrossoverTypesEnum.one_point,
    CrossoverTypesEnum.exchange_edges,
    CrossoverTypesEnum.exchange_parents_one,
    CrossoverTypesEnum.exchange_parents_both
]


@pytest.mark.parametrize('crossover_type', crossover_types)
@pytest.mark.parametrize('seed', [0, 42, 1042])
def test_crossover_reproducibility(crossover_type, seed):
    graph_example_first = graph_first()
    graph_example_second = graph_second()

    parameters = GPAlgorithmParameters()
    requirements = GraphRequirements()
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd'])
    crossover = Crossover(parameters, requirements, graph_generation_params)

    crossover.parameters.crossover_types = [crossover_type]

    def run_with_seed(seed):
        set_random_seed(seed)
        results = crossover([Individual(graph_example_first), Individual(graph_example_second)])
        results = [ind.graph.descriptive_id for ind in results]
        return results

    results_first = run_with_seed(seed)
    results_second = run_with_seed(seed)

    assert results_first == results_second


@pytest.mark.parametrize('mutation_type', MutationTypesEnum)
@pytest.mark.parametrize('seed', [0, 42, 1042])
def test_mutation_reproducibility(mutation_type, seed):
    params = get_mutation_params([mutation_type])
    mutation = Mutation(**params)

    def run_with_seed(seed):
        set_random_seed(seed)
        ind = Individual(graph_first())
        new_ind = mutation(ind)
        if isinstance(new_ind, Individual):
            return new_ind.graph.descriptive_id
        else:
            return new_ind

    results_first = run_with_seed(seed)
    results_second = run_with_seed(seed)

    assert results_first == results_second
