import os
import shutil
from datetime import timedelta
from functools import partial

import pytest

from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, SAVED_STATE_BASE_DIR
from golem.core.paths import default_data_dir
from golem.metrics.edit_distance import tree_edit_dist

NODE_TYPES = ('a', 'b')


@pytest.fixture
def saved_state_path():
    """Unique (per test) relative dir for state files; removes it afterwards."""
    path = os.path.join(SAVED_STATE_BASE_DIR, 'test_saved_state')
    full_path = os.path.join(default_data_dir(), path)
    shutil.rmtree(full_path, ignore_errors=True)
    yield path
    shutil.rmtree(full_path, ignore_errors=True)


def get_optimizer(num_of_generations: int, **saved_state_params) -> EvoGraphOptimizer:
    objective = Objective(partial(tree_edit_dist, generate_labeled_graph('tree', 8, NODE_TYPES)))
    requirements = GraphRequirements(timeout=timedelta(minutes=2),
                                     num_of_generations=num_of_generations,
                                     show_progress=False)
    gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter(), available_node_types=NODE_TYPES)
    algo_params = GPAlgorithmParameters(pop_size=10)
    initial_population = [generate_labeled_graph('tree', 4, NODE_TYPES) for _ in range(10)]
    return EvoGraphOptimizer(objective, initial_population, requirements, gen_params, algo_params,
                             **saved_state_params)


def get_objective() -> Objective:
    return Objective(partial(tree_edit_dist, generate_labeled_graph('tree', 8, NODE_TYPES)))


def test_saved_state(saved_state_path):
    num_of_generations_run_1 = 4
    num_of_generations_run_2 = 7

    # First optimizer: run to the end while saving the state after every generation (save_state_delta=0)
    optimiser1 = get_optimizer(num_of_generations_run_1, saved_state_path=saved_state_path)
    objective = get_objective()
    optimiser1.optimise(objective, save_state_delta=0)

    # Only the latest file is kept: older snapshots of the run must be cleaned up
    run_dir = os.path.join(default_data_dir(), saved_state_path, optimiser1._run_id)
    assert len(os.listdir(run_dir)) == 1, 'Wrong number of saved state files'

    # A second optimizer object is created deliberately: restoring the state into a fresh
    # instance is exactly the scenario the feature enables (continuation of the optimisation
    # after the original process has died)
    optimiser2 = get_optimizer(num_of_generations_run_2,
                               use_saved_state=True, saved_state_path=saved_state_path)

    # With save_state_delta=0 the last snapshot is written after the last evolutionary generation.
    # The only generation appended after that snapshot is the 'final_choices' bookkeeping generation,
    # so the restored generation number is exactly one less than the final one
    assert optimiser2.current_generation_num == optimiser1.current_generation_num - 1, \
        'Restored generation number does not correspond to the last saved state'
    # The restored state continues the same run: new snapshots go to the same run folder
    assert optimiser2._run_id == optimiser1._run_id
    assert optimiser2.best_individuals, 'Restored best_individuals are empty'
    assert optimiser2.population is not None, 'Restored population is empty'
    # The new timeout is reduced by the time the first run had already spent
    assert optimiser2.timer.timeout < timedelta(minutes=2), 'Timeout was not adjusted'
    assert optimiser2.generations.stagnation_iter_count <= optimiser1.generations.stagnation_iter_count

    restored_generation_num = optimiser2.current_generation_num
    optimiser2.optimise(objective, save_state_delta=0)

    # The second run continued from the restored generation (did not start from scratch)
    # and made it to its own num_of_generations limit: the optimisation stops when
    # current_generation_num reaches num_of_generations + 1 (the initial population counts
    # as the first generation) and then 'final_choices' adds one more
    assert restored_generation_num > 1
    assert optimiser2.current_generation_num == num_of_generations_run_2 + 2

    # The continued run must not be worse than the first one
    assert optimiser2.best_individuals[0].fitness.value <= optimiser1.best_individuals[0].fitness.value


def test_saved_state_fallback_to_scratch(saved_state_path):
    """If there is nothing to restore, a warning is logged and optimisation starts from scratch."""
    num_of_generations = 2
    optimiser = get_optimizer(num_of_generations,
                              use_saved_state=True, saved_state_path=saved_state_path)
    assert not optimiser._is_restored_from_saved_state

    found_graphs = optimiser.optimise(get_objective(), save_state_delta=1000)
    assert found_graphs
    assert optimiser.current_generation_num == num_of_generations + 2
