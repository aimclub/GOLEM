import time
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
        new_population = self._mutate_over_population(new_population, evaluator)
        self._check_final_population(new_population)
        return new_population

    def _mutate_over_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        # some params
        n_jobs = self.mutation.requirements.n_jobs
        target_pop_size = self.parameters.pop_size
        population_descriptive_ids_mapping = {ind.graph.descriptive_id: ind for ind in population}
        mutation_types = self.mutation._operator_agent.actions
        left_tries = [target_pop_size * MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER]
        mutations_per_individual = left_tries[0] / len(population)
        all_mutations_count_for_each_ind = {descriptive_id: 0
                                            for descriptive_id in population_descriptive_ids_mapping}
        mutation_count_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}

        # increase probability of mutation
        initial_parameters = deepcopy(self.parameters)
        initial_parameters.mutation_prob = 1.0
        self.mutation.update_requirements(parameters=initial_parameters)


        # additional functions
        def try_mutation(descriptive_id: str, mutation_type: Optional[MutationType] = None):
            left_tries[0] -= 1
            return executor.submit(self._mutation_n_evaluation, descriptive_id,
                                   population_descriptive_ids_mapping[descriptive_id],
                                   mutation_type, evaluator)

        def check_and_try_mutation(parent_descriptive_id: str,
                                   mutation_type: Optional[MutationType] = None,
                                   count: int = 1):
            # probs should be the same order as mutation_types
            probs = dict(zip(mutation_types, self.mutation._operator_agent.get_action_probs()))
            # check probability allows to make mutations
            if (probs[mutation_type] > (mutation_count_for_each_ind[parent_descriptive_id][mutation_type] /
                                        all_mutations_count_for_each_ind[parent_descriptive_id])):
                # check that there is not enough mutations
                if all_mutations_count_for_each_ind[parent_descriptive_id] < mutations_per_individual:
                    for _ in range(count):
                        try_mutation(parent_descriptive_id, mutation_type)
                    return True
            return False

        def add_new_individual_to_new_population(parent_descriptive_id: str,
                                                 mutation_type: MutationType,
                                                 new_individual: Individual):
            if new_individual:
                descriptive_id = new_individual.graph.descriptive_id
                if descriptive_id not in self._pop_graph_descriptive_ids:
                    mutation_count_for_each_ind[parent_descriptive_id][mutation_type] += 1
                    all_mutations_count_for_each_ind[parent_descriptive_id] += 1
                    new_population.append(new_individual)
                    self._pop_graph_descriptive_ids.add(descriptive_id)
                    return True
            return False

        # start reproducing
        new_population = []
        executor = get_reusable_executor(max_workers=n_jobs)

        # stage 1
        # set up each type of mutation for each individual
        futures = deque(try_mutation(descriptive_id, mutation_type)
                        for mutation_type in np.random.permutation(mutation_types)
                        for descriptive_id in np.random.permutation(list(population_descriptive_ids_mapping)))

        # stage 2
        delayed_mutations = deque()
        excessive_mutation_count = 0
        optimal_future_length = n_jobs + 4
        while futures:
            if len(new_population) == target_pop_size or left_tries[0] == 0:
                break

            # add new individual to new population
            parent_descriptive_id, mutation_type, new_ind = futures.popleft().result()
            added = add_new_individual_to_new_population(parent_descriptive_id, mutation_type, new_ind)

            # skip new mutation
            if added and excessive_mutation_count > 0 and len(futures) >= optimal_future_length:
                excessive_mutation_count -= 1
                continue

            # create new future with same mutation and same individual
            count = min(2, max(1, optimal_future_length - len(futures)))
            applied = check_and_try_mutation(parent_descriptive_id, mutation_type, count)
            if not applied or len(futures) < optimal_future_length:
                if len(futures) < optimal_future_length:
                    print(1)
                delayed_mutations.append((parent_descriptive_id, mutation_type))
                for _ in range(len(delayed_mutations) - 1):
                    parent_descriptive_id, mutation_type = delayed_mutations.popleft()
                    applied = check_and_try_mutation(parent_descriptive_id, mutation_type, count)
                    if applied: break
                    delayed_mutations.append((parent_descriptive_id, mutation_type))
            excessive_mutation_count += count - 1 if applied else 0

        # if there are any feature then process it and add new_ind to new_population if it is ready
        for future in futures:
            if future._state == 'FINISHED':
                add_new_individual_to_new_population(*future.result())
        executor.shutdown(wait=False)

        # Reset mutation probabilities to default
        self.mutation.update_requirements(requirements=self.parameters)
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
