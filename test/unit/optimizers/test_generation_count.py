import datetime

import pytest

from golem.core.adapter import DirectAdapter
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer
from test.unit.utils import graph_first, graph_second, graph_third, graph_fourth, graph_fifth

SETUP_LABELS = ('initial_assumptions', 'extended_initial_assumptions', 'final_choices')


def run_optimizer(n_initial: int, num_of_generations: int = 3, pop_size: int = 5):
    graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    objective = Objective({'graph_size': lambda graph: len(graph.nodes)})
    requirements = GraphRequirements(num_of_generations=num_of_generations,
                                     timeout=datetime.timedelta(minutes=5),
                                     max_depth=5, keep_history=True,
                                     early_stopping_iterations=None)
    graph_generation_params = GraphGenerationParams(adapter=DirectAdapter(),
                                                    rules_for_constraint=DEFAULT_DAG_RULES,
                                                    available_node_types=['a', 'b', 'c', 'd', 'e', 'f'])
    parameters = GPAlgorithmParameters(pop_size=pop_size, max_pop_size=pop_size,
                                       genetic_scheme_type=GeneticSchemeTypesEnum.generational)
    optimizer = EvoGraphOptimizer(objective, graphs[:n_initial], requirements,
                                  graph_generation_params, parameters)
    optimizer.optimise(ObjectiveEvaluate(objective))
    return optimizer


@pytest.mark.parametrize('n_initial', [5, 1])
def test_number_of_real_generations_matches_requirements(n_initial):
    """The requested number of generations must be spent on actual evolution steps
    whether or not the initial population needed an extension."""
    num_of_generations = 3
    optimizer = run_optimizer(n_initial, num_of_generations)
    evolution_generations = [generation for generation in optimizer.history.generations
                             if generation.label not in SETUP_LABELS]
    assert len(evolution_generations) == num_of_generations


def test_base_extend_population_copies_graphs():
    """Individuals produced by the extension must not share one mutable graph object."""
    source = [Individual(DirectAdapter().adapt(graph_first()))]
    extended = PopulationalOptimizer._extend_population(None, source, 5)
    assert len(extended) == 5
    graph_ids = [id(individual.graph) for individual in extended]
    assert len(set(graph_ids)) == len(graph_ids)
