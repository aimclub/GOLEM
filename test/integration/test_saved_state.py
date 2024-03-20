import os
import time
import glob

from datetime import timedelta
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.paths import default_data_dir
from golem.metrics.edit_distance import tree_edit_dist
from functools import partial


def find_latest_file_in_dir(directory: str) -> str:
    return max(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)

def test_saved_state():
    # Set params
    size = 16
    num_of_generations_run_1 = 40
    num_of_generations_run_2 = 45
    timeout = 10
    saved_state_path = 'saved_optimisation_state/test'

    # Generate target graph sought by optimizer using edit distance objective
    node_types = ('a', 'b')  # Available node types that can appear in graphs
    target_graph = generate_labeled_graph('tree', size, node_types)
    objective = Objective(partial(tree_edit_dist, target_graph))
    initial_population = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]

    # Setup optimization parameters
    requirements_run_1 = GraphRequirements(timeout=timedelta(minutes=timeout),
                                     num_of_generations=num_of_generations_run_1)
    requirements_run_2 = GraphRequirements(timeout=timedelta(minutes=timeout),
                                           num_of_generations=num_of_generations_run_2)

    gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter(), available_node_types=node_types)
    algo_params = GPAlgorithmParameters(pop_size=30)

    # Build and run the optimizer to create a saved state file
    optimiser1 = EvoGraphOptimizer(objective, initial_population, requirements_run_1, gen_params, algo_params,
                                  saved_state_path=saved_state_path)
    st = time.time()
    optimiser1.optimise(objective, save_state_delta=1)
    et = time.time()
    time1 = int(et - st) / 60

    # Check that the file with saved state was created
    saved_state_full_path = os.path.join(default_data_dir(), saved_state_path, optimiser1._run_id)
    saved_state_file = find_latest_file_in_dir(saved_state_full_path)
    assert os.path.isfile(saved_state_file) is True, 'ERROR: Saved state file was not created!'

    # Create the optimizer to check that the saved state was used
    optimiser2 = EvoGraphOptimizer(objective, initial_population, requirements_run_2, gen_params, algo_params,
                                  use_saved_state=True, saved_state_path=saved_state_path)

    # Check that the restored object has the same main parameters as the original or at least the params are not empty
    assert optimiser1.current_generation_num == optimiser2.current_generation_num + 1, \
        f'ERROR: Restored object field \'current_generation_num\' has wrong value: {optimiser2.current_generation_num}'
    assert optimiser1.generations.stagnation_iter_count == optimiser2.generations.stagnation_iter_count + 1, \
        f'ERROR: Restored object field \'generations.stagnation_iter_count\' has wrong value: ' \
        f'{optimiser2.generations.stagnation_iter_count}'
    assert optimiser1.best_individuals != [], f'ERROR: Restored object field \'best_individuals\' is empty'
    assert optimiser1.population is not None, f'ERROR: Restored object field \'population\' is empty'
    assert optimiser1.timer.timeout > optimiser2.timer.timeout, 'ERROR: timeout was not adjusted correctly'

    st = time.time()
    optimiser2.optimise(objective)
    et = time.time()
    time2 = int(et - st) / 60

    # Make sure the second run made it to the end
    assert optimiser2.current_generation_num == num_of_generations_run_2 + 2

    print(time1)
    print(time2)
    # Check that the run with the saved state takes less time than it would without it
    assert time1 > 2
    assert time2 < 1

    # Check that the result of the second optimisation is the same as or better than the first
    assert optimiser2.best_individuals[0].fitness.value <= optimiser1.best_individuals[0].fitness.value
