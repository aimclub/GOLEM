from collections import deque
from concurrent.futures import as_completed
from functools import partial
from itertools import cycle, chain
from math import ceil
from typing import Callable, Dict, Union, List, Optional
from multiprocessing import Queue, Manager
import queue
from copy import copy, deepcopy

import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.base_mutations import MutationTypesEnum
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationType
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.core.optimisers.opt_history_objects.individual import Individual


class ReproductionController:
    """
    Task of the Reproduction Controller is to reproduce population
    while keeping population size as specified in optimizer settings.

    Args:
        parameters: genetic algorithm parameters.
        selection: operator used in reproduction.
        mutation: operator used in reproduction.
        crossover: operator used in reproduction.
        window_size: size in iterations of the moving window to compute reproduction success rate.
    """

    def __init__(self,
                 parameters: GPAlgorithmParameters,
                 selection: Selection,
                 mutation: Mutation,
                 crossover: Crossover,
                 verifier: GraphVerifier):
        self.parameters = parameters
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover
        self.verifier = verifier

        self._pop_graph_descriptive_ids = set()
        self._minimum_valid_ratio = parameters.required_valid_ratio * 0.5

        self._log = default_log(self)

    def reproduce(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
           Implements additional checks on population to ensure that population size
           follows required population size.
        """
        selected_individuals = self.selection(population, self.parameters.pop_size)
        new_population = self.crossover(selected_individuals)

        # increase probability of mutation
        initial_parameters = deepcopy(self.parameters)
        initial_parameters.mutation_prob = 1.0
        self.mutation.update_requirements(parameters=initial_parameters)

        # create new populatin with mutations
        new_population = self._mutate_over_population(new_population, evaluator)

        # Reset mutation probabilities to default
        self.mutation.update_requirements(requirements=self.parameters)

        return new_population

    def _mutate_over_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        # some params
        n_jobs = self.mutation.requirements.n_jobs
        tasks_queue_length = n_jobs + 1
        target_pop_size = self.parameters.pop_size
        multiplier = target_pop_size / len(population)
        population_descriptive_ids_mapping = {ind.graph.descriptive_id: ind for ind in population}
        mutation_types = self.mutation._operator_agent.actions

        # counters and limits
        max_tries = target_pop_size * MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
        tries = [0, max_tries]  # [current count, max count]
        max_tries_for_each_mutation = 2 * ceil(max_tries / target_pop_size / len(mutation_types))
        mutation_count_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}
        all_mutations_count_for_each_ind = {descriptive_id: 0 for descriptive_id in population_descriptive_ids_mapping}
        mutation_tries_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}
        guessed_mutation_count = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                  for descriptive_id in population_descriptive_ids_mapping}

        # queues
        descriptive_id_queue = cycle(mutation_count_for_each_ind)
        mutation_type_queue = cycle(mutation_types)
        forbidden_mutations = {descriptive_id: set() for descriptive_id in population_descriptive_ids_mapping}

        def iterate_over_descriptive_ids(count: int = len(population_descriptive_ids_mapping)):
            for _ in range(count):
                yield next(descriptive_id_queue)

        def iterate_over_mutations(descriptive_id: str, count: int = len(mutation_types)):
            for _ in range(count):
                mutation_type = next(mutation_type_queue)
                if mutation_type not in forbidden_mutations[descriptive_id]:
                    yield mutation_type

        # additional functions
        def try_mutation(descriptive_id: str, mutation_type: Optional[MutationType] = None):
            tries[0] += 1
            mutation_tries_for_each_ind[descriptive_id][mutation_type] += 1
            return executor.submit(self._mutation_n_evaluation, descriptive_id,
                                   population_descriptive_ids_mapping[descriptive_id],
                                   mutation_type, evaluator)

        def add_new_individual_to_new_population(parent_descriptive_id, mutation_type, new_individual):
            if new_individual:
                descriptive_id = new_individual.graph.descriptive_id
                if descriptive_id not in self._pop_graph_descriptive_ids:
                    mutation_count_for_each_ind[parent_descriptive_id][mutation_type] += 1
                    all_mutations_count_for_each_ind[parent_descriptive_id] += 1
                    new_population.append(new_individual)
                    self._pop_graph_descriptive_ids.add(descriptive_id)
                    if len(new_population) == target_pop_size:
                        return True
            return False

        # start reproducing
        new_population = []
        executor = get_reusable_executor(max_workers=n_jobs)

        # stage 1
        # set up each type of mutation for each individual
        futures = deque
        for descriptive_id in np.random.permutation(list(population_descriptive_ids_mapping)):
            for mutation_type in np.random.permutation(mutation_types):
                futures.append(try_mutation(descriptive_id, mutation_type))
        # get some results from parallel computation for reducing calculation queue
        for future in as_completed(futures):
            population_is_prepared = add_new_individual_to_new_population(*future.result())
            if population_is_prepared: return new_population
            if len(futures) <= tasks_queue_length: break


        # stage 2
        # set up mutations until of all them will be applied once
        # if mutation does not work for some times, then do not use it

        while True:
            # get next finished future
            for _ in range(len(futures)):
                future = futures.popleft()
                if future._state == 'FINISHED': break
                futures.append(future)

            # add new individual to new population
            parent_descriptive_id, mutation_type, new_ind = future.result()
            population_is_prepared = add_new_individual_to_new_population(parent_descriptive_id, mutation_type, new_ind)
            if population_is_prepared: return new_population
            # if there are a lot of tasks go to next task
            if len(futures) > tasks_queue_length: continue

            # create new futures with mutation if mutation is not applied yet and not forbidden
            for descriptive_id in iterate_over_descriptive_ids():
                for mutation_type in iterate_over_mutations(descriptive_id):
                    if mutation_tries_for_each_ind[descriptive_id][mutation_type] < max_tries_for_each_mutation:
                        if mutation_count_for_each_ind[descriptive_id][mutation_type] == 0:
                            new_future = try_mutation(descriptive_id, mutation_type)
                            futures.append(new_future)
                    else:
                        # forbidd mutation if it not works
                        forbidden_mutations[descriptive_id].add(mutation_type)

        print(1)
        # check that forbidden_mutations works
        # check that as_completed(futures) works with list





        def get_next_parent_descriptive_id_with_next_mutation():
            min_mutation_count = min(all_mutations_count_for_each_ind.values())
            # for parent_descriptive_id, is_finished in finished_initial_individuals.items():
            for _ in range(len(population)):
                parent_descriptive_id = next(individuals_order)
                if all_mutations_count_for_each_ind[parent_descriptive_id] <= min_mutation_count:
                    for _ in range(len(mutation_types)):
                        mutation_type = next(mutations_order)
                        if mutation_tries_for_each_ind[parent_descriptive_id][mutation_type] == 0:
                            return parent_descriptive_id, mutation_type

                    # place for error if mutation_types order in _operator_agent and in mutation_types is differ
                    # mutations_shares = dict()
                    # all_tries = max(1, sum(mutation_tries_for_each_ind[parent_descriptive_id].values()))
                    # for mutation_type, prob in zip(mutation_types, self.mutation._operator_agent.get_action_probs()):
                    #     shares = (mutation_count_for_each_ind[parent_descriptive_id][mutation_type] /
                    #               max(1, multiplier * prob),
                    #               mutation_tries_for_each_ind[parent_descriptive_id][mutation_type] / all_tries)
                    #     if shares[0] < 1:
                    #         mutations_shares[mutation_type] = shares[0] * 2 + shares[1]
                    # return parent_descriptive_id, min(mutations_shares.items(), key=lambda x: x[1])[0]

                    guessed_counts = dict()
                    for mutation_type, prob in zip(mutation_types, self.mutation._operator_agent.get_action_probs()):
                        allowed_count = max(1, multiplier * prob)
                        if (allowed_count <= mutation_count_for_each_ind[parent_descriptive_id][mutation_type]):
                            successful_rate = (mutation_count_for_each_ind[parent_descriptive_id][mutation_type] /
                                               mutation_tries_for_each_ind[parent_descriptive_id][mutation_type])
                            guess_next_count = (mutation_count_for_each_ind[parent_descriptive_id][mutation_type] +
                                                successful_rate)
                            guessed_counts[mutation_type] = guess_next_count / allowed_count
                    return parent_descriptive_id, min(guessed_counts.items(), key=lambda x: x[1])[0]
            return None, None

        # set up some mutations for each ind
        results = deque(try_mutation(*args)
                        for args in (list(population_descriptive_ids_mapping.items())
                                     * self.mutation.requirements.n_jobs)[:self.mutation.requirements.n_jobs])
        new_population = list()
        print(f"{len(results)}")
        for _ in range(max_tries - len(results)):
            if len(new_population) >= target_pop_size or (not results):
                break

            parent_descriptive_id, mutation_type, new_ind = results.popleft().result()
            add_new_individual_to_new_population(new_ind)

            parent_descriptive_id, next_mutation = get_next_parent_descriptive_id_with_next_mutation()
            if next_mutation:
                print(f"next_mutation: {next_mutation} / {mutation_tries_for_each_ind[parent_descriptive_id][next_mutation]}")
                print(f"other mutations: {list(mutation_tries_for_each_ind[parent_descriptive_id].values())} / {list(mutation_count_for_each_ind[parent_descriptive_id].values())}")
                mutation_tries_for_each_ind[parent_descriptive_id][next_mutation] += 1
                new_res = try_mutation(parent_descriptive_id,
                                       population_descriptive_ids_mapping[parent_descriptive_id],
                                       next_mutation)
                results.append(new_res)

        # if there are any feature then process it and add new_ind to new_population if it is ready
        for future in results:
            if future._state == 'FINISHED':
                add_new_individual_to_new_population(future.result()[-1])
        executor.shutdown(wait=False)

        self._check_final_population(new_population)
        return new_population

    def _check_final_population(self, population: PopulationT) -> None:
        """ If population do not achieve required length return a warning or raise exception """
        target_pop_size = self.parameters.pop_size
        helpful_msg = ('Check objective, constraints and evo operators. '
                       'Possibly they return too few valid individuals.')

        if len(population) < target_pop_size * self._minimum_valid_ratio:
            raise EvaluationAttemptsError('Could not collect valid individuals'
                                          ' for population.' + helpful_msg)
        elif len(population) < target_pop_size:
            self._log.warning(f'Could not achieve required population size: '
                              f'have {len(population)},'
                              f' required {target_pop_size}!\n' + helpful_msg)

    def _mutation_n_evaluation(self,
                               descriptive_id: str,
                               individual: Individual,
                               mutation_type: Optional[MutationType],
                               evaluator: EvaluationOperator):
        individual, mutation_type, applied = self.mutation._mutation(individual, mutation_type=mutation_type)
        if applied and individual and self.verifier(individual.graph):
            individuals = evaluator([individual])
            if individuals:
                # if all is ok return all data
                return descriptive_id, mutation_type, individuals[0]
        # if something go wrong do not return new individual
        return descriptive_id, mutation_type, None
