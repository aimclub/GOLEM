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
        multiplier = target_pop_size / len(population)
        population_descriptive_ids_mapping = {ind.graph.descriptive_id: ind for ind in population}

        # mutations counters
        mutation_types = self.mutation._operator_agent.actions
        mutation_count_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}
        all_mutations_count_for_each_ind = {descriptive_id: 0 for descriptive_id in population_descriptive_ids_mapping}
        mutation_tries_for_each_ind = {descriptive_id: {mutation_type: 0 for mutation_type in mutation_types}
                                       for descriptive_id in population_descriptive_ids_mapping}
        individuals_order = cycle(mutation_count_for_each_ind)
        mutations_order = cycle(mutation_types)

        # increase probability of mutation
        initial_parameters = deepcopy(self.parameters)
        initial_parameters.mutation_prob = 1.0
        self.mutation.update_requirements(parameters=initial_parameters)

        executor = get_reusable_executor(max_workers=self.mutation.requirements.n_jobs)

        def try_mutation(descriptive_id: str, individual: Individual, mutation_type: Optional[MutationType] = None):
            return executor.submit(self._mutation_n_evaluation, descriptive_id, individual, mutation_type, evaluator)

        def add_new_individual_to_new_population(new_individual):
            if new_individual:
                descriptive_id = new_individual.graph.descriptive_id
                if descriptive_id not in self._pop_graph_descriptive_ids:
                    mutation_count_for_each_ind[parent_descriptive_id][mutation_type] += 1
                    all_mutations_count_for_each_ind[parent_descriptive_id] += 1
                    new_population.append(new_individual)
                    self._pop_graph_descriptive_ids.add(descriptive_id)

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
