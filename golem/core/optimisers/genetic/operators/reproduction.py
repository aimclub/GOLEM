import time
from collections import deque
from concurrent.futures import as_completed
from functools import partial
from itertools import cycle, chain
from math import ceil
from multiprocessing.managers import ValueProxy, DictProxy
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
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationType, SpecialSingleMutation
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
        with (Manager() as manager,
              Parallel(n_jobs=self.mutation.requirements.n_jobs, return_as='generator') as parallel):

            operator_agent = manager.Value('operator_agent', self.mutation._operator_agent)
            agent_experience = manager.Value('agent_experience', self.mutation.agent_experience)
            mutation = SpecialSingleMutation(parameters=self.mutation.parameters,
                                             requirements=self.mutation.requirements,
                                             graph_gen_params=self.mutation.graph_generation_params,
                                             mutations_repo=self.mutation._mutations_repo,
                                             operator_agent=operator_agent,
                                             agent_experience=agent_experience)
            pop_graph_descriptive_ids = manager.dict(zip(self._pop_graph_descriptive_ids,
                                                         range(len(self._pop_graph_descriptive_ids))))
            mutation_fun = partial(self._mutation_n_evaluation,
                                   count=self.parameters.max_num_of_mutation_attempts,
                                   pop_graph_descriptive_ids=pop_graph_descriptive_ids,
                                   mutation=mutation,
                                   evaluator=evaluator)

            max_tries = self.parameters.pop_size * MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
            new_population = []

            new_ind_generator = parallel(delayed(mutation_fun)(ind)
                                         for ind, _ in zip(cycle(population), range(max_tries)))
            for new_ind in new_ind_generator:
                if new_ind:
                    new_population.append(new_ind)
                    if len(new_population) >= self.parameters.pop_size:
                        break
            self._pop_graph_descriptive_ids |= set(pop_graph_descriptive_ids)
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
                               individual: Individual,
                               count: int,
                               pop_graph_descriptive_ids: DictProxy,
                               mutation: SpecialSingleMutation,
                               evaluator: EvaluationOperator):
        origin, mutation_type = individual, None
        for _ in range(count):
            individual, mutation_type = mutation(origin, mutation_type=mutation_type)
            if individual and self.verifier(individual.graph):
                descriptive_id = individual.graph.descriptive_id
                if descriptive_id not in pop_graph_descriptive_ids:
                    pop_graph_descriptive_ids[descriptive_id] = True
                    individuals = evaluator([individual])
                    if individuals:
                        return individuals[0]
