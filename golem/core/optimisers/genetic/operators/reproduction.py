ctions import deque
from functools import partial
from itertools import cycle, chain
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
        return new_population

    def _mutate_over_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        target_pop_size = self.parameters.pop_size
        max_tries = target_pop_size * MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
        max_attempts_count = self.parameters.max_num_of_operator_attempts
        multiplier = target_pop_size / len(population)
        population_descriptive_ids_mapping = {ind.graph.descriptive_id: ind for ind in population}
        finished_initial_individuals = {descriptive_id: False for descriptive_id in population_descriptive_ids_mapping}

        # mutations counters
        mutation_types = self.mutation._operator_agent.actions
        mutation_count_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}
        all_mutations_count_for_each_ind = {descriptive_id: 0 for descriptive_id in population_descriptive_ids_mapping}
        mutation_tries_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}

        # increase probability of mutation
        initial_parameters = deepcopy(self.parameters)
        initial_parameters.mutation_prob = 1.0
        self.mutation.update_requirements(parameters=initial_parameters)

        executor = get_reusable_executor(max_workers=self.mutation.requirements.n_jobs)

        def try_mutation(descriptive_id: str, individual: Individual, mutation_type: Optional[MutationType] = None):
            return executor.submit(self._mutation_n_evaluation, descriptive_id, individual, mutation_type, evaluator)

        def add_new_individual_to_new_population(new_individual):
            mutation_tries_for_each_ind[parent_descriptive_id][mutation_type] += 1
            if new_individual:
                descriptive_id = new_individual.graph.descriptive_id
                if descriptive_id not in self._pop_graph_descriptive_ids:
                    mutation_count_for_each_ind[parent_descriptive_id][mutation_type] += 1
                    all_mutations_count_for_each_ind[parent_descriptive_id] += 1
                    new_population.append(new_individual)
                    self._pop_graph_descriptive_ids.add(descriptive_id)

        def get_next_parent_descriptive_id_with_allowed_operations():
            for parent_descriptive_id, is_finished in finished_initial_individuals.items():
                if not is_finished:
                    if all_mutations_count_for_each_ind[parent_descriptive_id] == 0:
                        # if there are no mutations then make any mutation
                        allowed_mutation_types = mutation_types
                    else:
                        # filter mutations for individual, rely on probabilities and mutation_count
                        # place for error if mutation_types order in _operator_agent and in mutation_types is differ
                        allowed_mutation_types = []
                        mutation_probabilities = self.mutation._operator_agent.get_action_probs()
                        allowed_mutations_count = [max(1, round(multiplier * x)) for x in mutation_probabilities]
                        for mutation_type, mutation_probability, allowed_count in zip(mutation_types,
                                                                                      mutation_probabilities,
                                                                                      allowed_mutations_count):
                            if allowed_count > mutation_count_for_each_ind[parent_descriptive_id][mutation_type]:
                                real_prob = (mutation_count_for_each_ind[parent_descriptive_id][mutation_type] /
                                             all_mutations_count_for_each_ind[parent_descriptive_id])
                                if real_prob < mutation_probability:
                                    allowed_mutation_types.append(mutation_type)
                    if not allowed_mutation_types:
                        finished_initial_individuals[parent_descriptive_id] = True
                    return parent_descriptive_id, allowed_mutation_types
            return None, None

        # set up len(mutation_types) // 2 evaluations for each ind
        results = deque(try_mutation(*args)
                        for args in list(population_descriptive_ids_mapping.items()) * int(len(mutation_types) // 2))
        new_population = list()
        print(f"{len(results)}")
        for _ in range(max_tries - len(results)):
            if len(new_population) >= target_pop_size or (not results):
                break

            parent_descriptive_id, mutation_type, new_ind = results.popleft().result()
            add_new_individual_to_new_population(new_ind)

            parent_descriptive_id, allowed_mutation_types = get_next_parent_descriptive_id_with_allowed_operations()
            if allowed_mutation_types:
                # choose next mutation with lowest tries count and run it
                next_mutation_type = min(allowed_mutation_types,
                                         key=lambda mutation_type:
                                         mutation_tries_for_each_ind[parent_descriptive_id][mutation_type])
                new_res = try_mutation(parent_descriptive_id,
                                       population_descriptive_ids_mapping[parent_descriptive_id],
                                       next_mutation_type)
                results.append(new_res)
            print(f"{len(results)}: {sum(future._state == 'FINISHED' for future in results)}")

        # if there are any feature then process it and add new_ind to new_population if it is ready
        for future in results:
            if future._state == 'FINISHED':
                add_new_individual_to_new_population(future.result()[-1])
        executor.shutdown(wait=False)

        # Reset mutation probabilities to default
        self.mutation.update_requirements(requirements=self.parameters)

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
