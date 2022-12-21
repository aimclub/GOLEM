import itertools
import os
from pathlib import Path

import numpy as np
import pytest

from golem.core.optimisers.fitness.fitness import SingleObjFitness
from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, Crossover
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum, Mutation
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.visualisation.opt_viz import PlotTypesEnum, OptHistoryVisualizer
from test.unit.adapter.mock_adapter import MockAdapter, MockDomainStructure, MockNode, MockObjectiveEvaluate
from test.unit.utils import RandomMetric, graph_first, graph_second, graph_third, graph_fourth, graph_fifth


def create_mock_graph_individual():
    node_1 = MockNode(content={'name': 'logit'})
    node_2 = MockNode(content={'name': 'lda'})
    node_3 = MockNode(content={'name': 'knn'})
    mock_graph = MockDomainStructure([node_1, node_2, node_3])
    individual = Individual(graph=mock_graph)
    individual.set_evaluation_result(SingleObjFitness(1))
    return individual


def create_individual(evaluated=True):
    first = OptNode(content={'name': 'logit'})
    second = OptNode(content={'name': 'lda'})
    final = OptNode(content={'name': 'knn'},
                    nodes_from=[first, second])

    individual = Individual(graph=OptGraph(final))
    if evaluated:
        individual.set_evaluation_result(SingleObjFitness(1))
    return individual


@pytest.fixture(scope='module')
def generate_history(request) -> OptHistory:
    generations_quantity, pop_size, ind_creation_func = request.param
    history = OptHistory()
    for gen_num in range(generations_quantity):
        new_pop = []
        for _ in range(pop_size):
            ind = ind_creation_func()
            ind.set_native_generation(gen_num)
            new_pop.append(ind)
        history.add_to_history(new_pop)
    return history


def _test_individuals_in_history(history: OptHistory):
    uids = set()
    ids = set()
    for ind in itertools.chain(*history.individuals):
        # All individuals in `history.individuals` must have a native generation.
        assert ind.has_native_generation
        assert ind.fitness
        if ind.native_generation == 0:
            continue
        # All individuals must have parents, except for the initial assumptions.
        assert ind.parents
        assert ind.parents_from_prev_generation
        # The first of `operators_from_prev_generation` must point to `parents_from_prev_generation`.
        assert ind.parents_from_prev_generation == list(ind.operators_from_prev_generation[0].parent_individuals)
        # All parents are from previous generations
        assert all(p.native_generation < ind.native_generation for p in ind.parents_from_prev_generation)

        uids.add(ind.uid)
        ids.add(id(ind))
        for parent_operator in ind.operators_from_prev_generation:
            uids.update({i.uid for i in parent_operator.parent_individuals})
            ids.update({id(i) for i in parent_operator.parent_individuals})

    assert len(uids) == len(ids)


@pytest.mark.parametrize('generate_history', [[2, 10, create_individual],
                                              [2, 10, create_mock_graph_individual]],
                         indirect=True)
def test_history_adding(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history

    assert len(history.individuals) == generations_quantity
    for gen in range(generations_quantity):
        assert len(history.individuals[gen]) == pop_size


@pytest.mark.parametrize('generate_history', [[2, 10, create_individual]], indirect=True)
def test_individual_graph_type_is_optgraph(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history
    for gen in range(generations_quantity):
        for ind in range(pop_size):
            assert type(history.individuals[gen][ind].graph) == OptGraph


def test_ancestor_for_crossover():
    adapter = MockAdapter()
    parent_ind_first = Individual(adapter.adapt(MockDomainStructure([MockNode('a')])))
    parent_ind_second = Individual(adapter.adapt(MockDomainStructure([MockNode('b')])))

    requirements = GraphRequirements()
    graph_params = GraphGenerationParams(available_node_types=['a', 'b'])
    opt_parameters = GPAlgorithmParameters(crossover_types=[CrossoverTypesEnum.subtree], crossover_prob=1)
    crossover = Crossover(opt_parameters, requirements, graph_params)
    crossover_results = crossover([parent_ind_first, parent_ind_second])

    for crossover_result in crossover_results:
        assert crossover_result.parent_operator
        assert crossover_result.parent_operator.type_ == 'crossover'
        assert len(crossover_result.parents) == 2
        assert crossover_result.parents[0].uid == parent_ind_first.uid
        assert crossover_result.parents[1].uid == parent_ind_second.uid


def test_ancestor_for_mutation():
    graph = MockDomainStructure([MockNode('a')])
    adapter = MockAdapter()
    parent_ind = Individual(adapter.adapt(graph))

    requirements = GraphRequirements()
    graph_params = GraphGenerationParams(available_node_types=['a'])
    parameters = GPAlgorithmParameters(mutation_types=[MutationTypesEnum.simple], mutation_prob=1)
    mutation = Mutation(parameters, requirements, graph_params)

    mutation_result = mutation(parent_ind)

    assert mutation_result.parent_operator
    assert mutation_result.parent_operator.type_ == 'mutation'
    assert len(mutation_result.parents) == 1
    assert mutation_result.parents[0].uid == parent_ind.uid


def test_parent_operator():
    graph = MockDomainStructure([MockNode('a')])
    adapter = MockAdapter()
    ind = Individual(adapter.adapt(graph))
    mutation_type = MutationTypesEnum.simple
    operator_for_history = ParentOperator(type_='mutation',
                                          operators=str(mutation_type),
                                          parent_individuals=ind)

    assert operator_for_history.parent_individuals[0] == ind
    assert operator_for_history.type_ == 'mutation'


@pytest.mark.parametrize('generate_history', [[2, 10, create_mock_graph_individual]], indirect=True)
def test_history_properties(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history
    assert len(history.all_historical_quality()) == pop_size * generations_quantity
    assert len(history.historical_fitness) == generations_quantity
    assert len(history.historical_fitness[0]) == pop_size
    assert len(history.all_historical_fitness) == pop_size * generations_quantity


def test_history_save_custom_nodedata():
    contents = [{'name': f'custom_{i}',
                 'important_field': ['secret', 42],
                 'matrix': np.random.randint(0, 100, (4 + 2 * i, 2 + i)).tolist()}
                for i in range(10)]

    graphs = [Individual(OptGraph(OptNode(content=content)), native_generation=i)
              for i, content in enumerate(contents)]

    history = OptHistory()
    history.add_to_history(graphs[:3])
    history.add_to_history(graphs[3:6])
    history.add_to_history(graphs[6:])

    saved = history.save()
    reloaded = OptHistory.load(saved)
    reloaded_inds = list(itertools.chain(*reloaded.individuals))

    for i, ind in enumerate(reloaded_inds):
        ind_content = ind.graph.root_node.content
        assert ind_content == contents[i]
        assert ind_content['matrix'] == contents[i]['matrix']


@pytest.mark.parametrize('generate_history', [[2, 10, create_individual]], indirect=True)
def test_prepare_for_visualisation(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history
    assert len(history.all_historical_fitness) == pop_size * generations_quantity

    leaderboard = history.get_leaderboard()
    assert OptNode('lda').descriptive_id in leaderboard
    assert 'Position' in leaderboard

    dumped_history = history.save()
    loaded_history = OptHistory.load(dumped_history)
    leaderboard = loaded_history.get_leaderboard()
    assert OptNode('lda').descriptive_id in leaderboard
    assert 'Position' in leaderboard


@pytest.mark.parametrize('generate_history', [[3, 4, create_individual]], indirect=True)
def test_all_historical_quality(generate_history):
    history = generate_history
    eval_fitness = [[0.9, 0.8], [0.8, 0.6], [0.2, 0.4], [0.9, 0.9]]
    weights = (-1, 1)
    for pop_num, population in enumerate(history.individuals):
        if pop_num != 0:
            eval_fitness = [[fit[0] + 0.5, fit[1]] for fit in eval_fitness]
        for ind_num, individual in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[ind_num], weights=weights)
            object.__setattr__(individual, 'fitness', fitness)
    all_quality = history.all_historical_quality()
    assert all_quality[0] == -0.9 and all_quality[4] == -1.4 and all_quality[5] == -1.3 and all_quality[10] == -1.2


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_newly_generated_history(n_jobs: int):
    num_of_gens = 5
    objective = Objective({'random_metric': RandomMetric.get_value})
    init_graphs = [graph_first(), graph_second(), graph_third(), graph_fourth(), graph_fifth()]
    requirements = GraphRequirements(num_of_generations=num_of_gens)
    graph_generation_params = GraphGenerationParams(available_node_types=['a', 'b', 'c', 'd', 'e', 'f'])
    opt_params = GPAlgorithmParameters(pop_size=5)
    opt = EvoGraphOptimizer(objective, init_graphs, requirements, graph_generation_params, opt_params)
    obj_eval = ObjectiveEvaluate(objective)
    opt.optimise(obj_eval)
    history = opt.history

    assert history is not None
    assert len(history.individuals) == num_of_gens + 2  # initial_assumptions + num_of_gens + final_choices
    assert len(history.archive_history) == num_of_gens + 2  # initial_assumptions + num_of_gens + final_choices
    assert len(history.initial_assumptions) == 5
    assert len(history.final_choices) == 1
    _test_individuals_in_history(history)
    # Test history dumps
    dumped_history_json = history.save()
    loaded_history = OptHistory.load(dumped_history_json)
    assert dumped_history_json is not None
    assert dumped_history_json == loaded_history.save(), 'The history is not equal to itself after reloading!'
    _test_individuals_in_history(loaded_history)


@pytest.mark.parametrize('generate_history', [[3, 4, create_individual],
                                              [3, 4, create_mock_graph_individual]],
                         indirect=True)
@pytest.mark.parametrize('plot_type', PlotTypesEnum)
def test_history_show_saving_plots(tmp_path, plot_type: PlotTypesEnum, generate_history):
    save_path = Path(tmp_path, plot_type.name)
    save_path = save_path.with_suffix('.gif') if plot_type is PlotTypesEnum.operations_animated_bar \
        else save_path.with_suffix('.png')
    history: OptHistory = generate_history
    visualizer = OptHistoryVisualizer(history)
    visualization = plot_type.value(visualizer.history, visualizer.visuals_params)
    visualization.visualize(save_path=str(save_path), best_fraction=0.1, dpi=100)
    if plot_type is not PlotTypesEnum.fitness_line_interactive:
        assert save_path.exists()


def test_history_correct_serialization():
    test_history_path = Path(__file__).parent.parent.parent
    test_history_path = os.path.join(test_history_path, 'data', 'test_history.json')

    history = OptHistory.load(test_history_path)
    dumped_history_json = history.save()
    reloaded_history = OptHistory.load(dumped_history_json)

    assert history.individuals == reloaded_history.individuals
    assert dumped_history_json == reloaded_history.save(), 'The history is not equal to itself after reloading!'
    _test_individuals_in_history(reloaded_history)


def test_collect_intermediate_metric():
    metric = RandomMetric.get_value
    graph_gen_params = GraphGenerationParams(available_node_types=['a', 'b', 'c'],
                                             adapter=MockAdapter())

    objective_eval = MockObjectiveEvaluate(Objective({'rand_metric': metric}))
    dispatcher = MultiprocessingDispatcher(graph_gen_params.adapter)
    dispatcher.set_evaluation_callback(objective_eval.evaluate_intermediate_metrics)
    evaluate = dispatcher.dispatch(objective_eval)

    population = [create_individual(evaluated=False)]
    evaluated_pipeline = evaluate(population)[0].graph
    restored_pipeline = graph_gen_params.adapter.restore(evaluated_pipeline)

    assert_intermediate_metrics(restored_pipeline)


def assert_intermediate_metrics(graph: MockDomainStructure):
    seen_metrics = []
    for node in graph.nodes:
        assert node.content['intermediate_metric'] is not None
        assert node.content['intermediate_metric'] not in seen_metrics
        seen_metrics.append(node.content['intermediate_metric'])
