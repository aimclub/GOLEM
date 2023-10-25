import itertools
import os
from pathlib import Path
from random import random

import numpy as np
import pytest
from hyperopt import hp

from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum, Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.paths import project_root
from golem.core.tuning.search_space import SearchSpace
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.visualisation.opt_viz import PlotTypesEnum, OptHistoryVisualizer
from golem.visualisation.opt_viz_extra import OptHistoryExtraVisualizer
from test.unit.mocks.common_mocks import MockAdapter, MockDomainStructure, MockNode, MockObjectiveEvaluate
from test.unit.utils import RandomMetric, graph_first, graph_second, graph_third, graph_fourth, graph_fifth


def create_mock_graph_individual():
    node_1 = MockNode(content={'name': 'logit'})
    node_2 = MockNode(content={'name': 'lda'})
    node_3 = MockNode(content={'name': 'knn'})
    mock_graph = MockDomainStructure([node_1, node_2, node_3])
    individual = Individual(graph=mock_graph)
    individual.set_evaluation_result(MultiObjFitness([random(), random()]))
    return individual


def create_individual(evaluated=True):
    first = OptNode(content={'name': 'logit'})
    second = OptNode(content={'name': 'lda'})
    final = OptNode(content={'name': 'knn'},
                    nodes_from=[first, second])

    individual = Individual(graph=OptGraph(final))
    if evaluated:
        individual.set_evaluation_result(MultiObjFitness([random(), random()]))
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
        # since only n best individuals need to be added to archive history
        history.add_to_evolution_best_archive(
            [sorted(new_pop, key=lambda ind: ind.fitness.values[0], reverse=False)[0]])
    return history


def test_individuals_in_history(history: OptHistory):
    uids = set()
    ids = set()
    for ind in itertools.chain(*history.generations):
        # All individuals in `history.generations` must have a native generation.
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

    assert len(history.generations) == generations_quantity
    for gen in range(generations_quantity):
        assert len(history.generations[gen]) == pop_size


@pytest.mark.parametrize('generate_history', [[2, 10, create_individual]], indirect=True)
def test_individual_graph_type_is_optgraph(generate_history):
    generations_quantity = 2
    pop_size = 10
    history = generate_history
    for gen in range(generations_quantity):
        for ind in range(pop_size):
            assert type(history.generations[gen][ind].graph) == OptGraph


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
    parameters = GPAlgorithmParameters(mutation_types=[MutationTypesEnum.single_add], mutation_prob=1)
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
    reloaded_inds = list(itertools.chain(*reloaded.generations))

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
    for pop_num, population in enumerate(history.generations):
        if pop_num != 0:
            eval_fitness = [[fit[0] + 0.5, fit[1]] for fit in eval_fitness]
        for ind_num, individual in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[ind_num], weights=weights)
            object.__setattr__(individual, 'fitness', fitness)
    all_quality = history.all_historical_quality()
    assert all_quality[0] == -0.9 and all_quality[4] == -1.4 and all_quality[5] == -1.3 and all_quality[10] == -1.2


@pytest.fixture()
def search_space():
    params_per_operation = {
        'a': {
            'a1': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 7],
                'type': 'discrete'
            },
            'a2': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-3, 1],
                'type': 'continuous'
            }
        },
        'b': {
            'b1': {
                'hyperopt-dist': hp.choice,
                'sampling-scope': [["first", "second", "third"]],
                'type': 'categorical'
            },
            'b2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
        },
        'e': {
            'e1': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            },
            'e2': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'
            }
        },
        'f': {
            'f': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-2, 10.0],
                'type': 'continuous'
            }
        }}
    return SearchSpace(params_per_operation)


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_newly_generated_history(n_jobs: int, search_space):
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

    tuning_iterations = 2
    tuner = SimultaneousTuner(obj_eval, search_space, MockAdapter(), iterations=tuning_iterations, n_jobs=n_jobs,
                              history=history)
    tuner.tune(history.evolution_results[0].graph)

    # initial_assumptions=1 + num_of_gens=5 + evolution_results=1 + tuning_start=1 +
    #   tuning_iterations=2 + tuning_result=1 -> num_of_gens + tuning_iterations + 4
    expected_gen_num = num_of_gens + tuning_iterations + 4

    # initial_assumptions=1 + num_of_gens=5 + evolution_results=1 -> num_of_gens + 2
    expected_evolution_gen_num = num_of_gens + 2

    assert history is not None
    assert len(history.generations) == expected_gen_num
    assert len(history.evolution_best_archive) == expected_evolution_gen_num
    assert len(history.initial_assumptions) == 5
    assert len(history.evolution_results) == 1
    assert len(history.tuning_result) == 1
    assert hasattr(history, 'objective')
    test_individuals_in_history(history)
    # Test history dumps
    dumped_history_json = history.save()
    loaded_history = OptHistory.load(dumped_history_json)
    assert dumped_history_json is not None
    assert dumped_history_json == loaded_history.save(), 'The history is not equal to itself after reloading!'
    test_individuals_in_history(loaded_history)


@pytest.mark.parametrize('generate_history', [[3, 4, create_individual],
                                              [3, 4, create_mock_graph_individual]],
                         indirect=True)
@pytest.mark.parametrize('plot_type', PlotTypesEnum)
def test_history_show_saving_plots(tmp_path, plot_type: PlotTypesEnum, generate_history):
    save_path = Path(tmp_path, plot_type.name)
    gif_plots = [PlotTypesEnum.operations_animated_bar,
                 PlotTypesEnum.diversity_population]
    save_path = save_path.with_suffix('.gif') if plot_type in gif_plots \
        else save_path.with_suffix('.png')
    history: OptHistory = generate_history
    visualizer = OptHistoryVisualizer(history)
    visualization = plot_type.value(visualizer.history, visualizer.visuals_params)
    visualization.visualize(save_path=str(save_path), best_fraction=0.1, dpi=100)
    if plot_type is not PlotTypesEnum.fitness_line_interactive:
        assert save_path.exists()


@pytest.mark.parametrize('generate_history', [[3, 4, create_individual],
                                              [3, 4, create_mock_graph_individual]],
                         indirect=True)
def test_extra_history_visualizer(tmp_path, generate_history):
    history: OptHistory = generate_history
    visualizer = OptHistoryExtraVisualizer(history, str(tmp_path))
    visualizer.visualise_history()
    visualizer.pareto_gif_create()
    visualizer.boxplots_gif_create()
    assert len(os.listdir(os.path.join(str(tmp_path), 'composing_history'))) == 3


def test_history_correct_serialization():
    test_history_path = Path(__file__).parent.parent.parent
    test_history_path = os.path.join(test_history_path, 'data', 'test_history.json')

    history = OptHistory.load(test_history_path)
    dumped_history_json = history.save()
    reloaded_history = OptHistory.load(dumped_history_json)

    assert history.generations == reloaded_history.generations
    assert dumped_history_json == reloaded_history.save(), 'The history is not equal to itself after reloading!'
    test_individuals_in_history(reloaded_history)


def test_collect_intermediate_metric():
    metric = RandomMetric.get_value
    graph_gen_params = GraphGenerationParams(available_node_types=['a', 'b', 'c'],
                                             adapter=MockAdapter())

    objective_eval = MockObjectiveEvaluate(Objective({'rand_metric': metric}))
    dispatcher = MultiprocessingDispatcher(graph_gen_params.adapter)
    dispatcher.set_graph_evaluation_callback(objective_eval.evaluate_intermediate_metrics)
    evaluate = dispatcher.dispatch(objective_eval)

    population = [create_individual(evaluated=False)]
    evaluated_graph = evaluate(population)[0].graph
    restored_graph = graph_gen_params.adapter.restore(evaluated_graph)

    assert_intermediate_metrics(restored_graph)


def test_load_zero_generations_history():
    """ Test to load histories with zero generations, since it still can contain info about
    objective, tuning result, etc. """
    path_to_history = os.path.join(project_root(), 'test', 'data', 'zero_gen_history.json')
    history = OptHistory.load(path_to_history)
    assert isinstance(history, OptHistory)
    assert len(history.evolution_best_archive) == 0
    assert history.objective is not None


@pytest.mark.parametrize('generate_history', [[100, 100, create_individual]], indirect=True)
def test_save_load_light_history(generate_history):
    history = generate_history
    file_name = 'light_history.json'
    path_to_dir = os.path.join(project_root(), 'test', 'data')
    path_to_history = os.path.join(path_to_dir, file_name)
    history.save(json_file_path=path_to_history, is_save_light=True)
    assert file_name in os.listdir(path_to_dir)
    loaded_history = OptHistory().load(path_to_history)
    assert isinstance(loaded_history, OptHistory)
    assert len(loaded_history.evolution_best_archive) == len(loaded_history.generations) == 100
    for i, _ in enumerate(loaded_history.generations):
        assert len(loaded_history.generations[i]) == len(loaded_history.evolution_best_archive[i]) == 1
    os.remove(path=os.path.join(path_to_dir, file_name))


@pytest.mark.parametrize('generate_history', [[50, 30, create_individual]], indirect=True)
def test_light_history_is_significantly_lighter(generate_history):
    """ Checks if light version of history weights signif """
    history = generate_history
    file_name_light = 'light_history.json'
    file_name_heavy = 'heavy_history.json'
    path_to_dir = os.path.join(project_root(), 'test', 'data')
    history.save(json_file_path=os.path.join(path_to_dir, file_name_light), is_save_light=True)
    history.save(json_file_path=os.path.join(path_to_dir, file_name_heavy), is_save_light=False)
    light_history_size = os.stat(os.path.join(path_to_dir, file_name_light)).st_size
    heavy_history_size = os.stat(os.path.join(path_to_dir, file_name_heavy)).st_size
    assert light_history_size * 25 <= heavy_history_size
    os.remove(path=os.path.join(path_to_dir, file_name_light))
    os.remove(path=os.path.join(path_to_dir, file_name_heavy))


def assert_intermediate_metrics(graph: MockDomainStructure):
    seen_metrics = []
    for node in graph.nodes:
        assert node.content['intermediate_metric'] is not None
        assert node.content['intermediate_metric'] not in seen_metrics
        seen_metrics.append(node.content['intermediate_metric'])
