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
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationType, SinglePredefinedMutation
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
        """
        selected_individuals = self.selection(population, self.parameters.pop_size)
        new_population = self.crossover(selected_individuals)
        new_population = self._mutate_over_population(new_population, evaluator)
        self._check_final_population(new_population)
        return new_population

    def _mutate_over_population(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """Generate new individuals by mutation in parallel.
           Implements additional checks on population to ensure that population size follows
           required population size. Also controls uniqueness of population.
        """
        with Manager() as manager:
            mutation = SinglePredefinedMutation(parameters=self.mutation.parameters,
                                                requirements=self.mutation.requirements,
                                                graph_gen_params=self.mutation.graph_generation_params,
                                                mutations_repo=self.mutation._mutations_repo)
            pop_graph_descriptive_ids = manager.dict(zip(self._pop_graph_descriptive_ids,
                                                         range(len(self._pop_graph_descriptive_ids))))

            left_tries = self.parameters.pop_size * MAX_GRAPH_GEN_ATTEMPTS_AS_POP_SIZE_MULTIPLIER
            executor = get_reusable_executor(max_workers=self.mutation.requirements.n_jobs)
            cycled_population = cycle(population)
            new_population = []
            futures = deque()

            def try_mutation(ind, mutation_type=None, count=self.parameters.max_num_of_mutation_attempts):
                mutation_type = mutation_type or self.mutation.agent.choose_action(ind.graph)
                return executor.submit(self._mutation_n_evaluation,
                                       individual=ind,
                                       count=count,
                                       mutation_type=mutation_type,
                                       pop_graph_descriptive_ids=pop_graph_descriptive_ids,
                                       mutation=mutation,
                                       evaluator=evaluator)

            while left_tries > 0:
                # create new tasks if there is not enough load
                if len(futures) < self.mutation.requirements.n_jobs + 2:
                    futures.append(try_mutation(next(cycled_population)))
                    continue

                # get next finished future
                while True:
                    future = futures.popleft()
                    if future._state == 'FINISHED': break
                    futures.append(future)
                    time.sleep(0.01)  # to prevent flooding

                # process result
                left_tries -= 1
                failed_stage, individual, mutation_type, retained_count = future.result()
                if failed_stage is None:
                    new_population.append(individual)
                    if len(new_population) >= self.parameters.pop_size:
                        break
                else:
                    if failed_stage == 'verification':
                        # add experience to mutation
                        self.mutation.agent_experience.collect_experience(individual, mutation_type, reward=-1.0)
                    if retained_count > 0:
                        futures.append(try_mutation(individual, mutation_type, retained_count))

            # get finished mutations
            for future in futures:
                if future._state == 'FINISHED':
                    applied, ind, *_ = future.result()
                    if applied: new_population.append(ind)

            # shutdown workers and add pop_graph_descriptive_ids to self._pop_graph_descriptive_ids
            executor.shutdown(wait=False)
            for _ in range(10):
                try:
                    self._pop_graph_descriptive_ids |= set(pop_graph_descriptive_ids)
                    break
                except RuntimeError as exception:
                    time.sleep(0.1)  # time for finish all processes
            else:
                raise exception

            return new_population

    def _mutation_n_evaluation(self,
                               individual: Individual,
                               count: int,
                               mutation_type: MutationType,
                               pop_graph_descriptive_ids: DictProxy,
                               mutation: SinglePredefinedMutation,
                               evaluator: EvaluationOperator):
        # mutation
        new_ind, mutation_type = mutation(individual, mutation_type=mutation_type)
        if not new_ind:
            return 'mutation', individual, mutation_type, count - 1

        # verification
        if not self.verifier(new_ind.graph):
            return 'verification', individual, mutation_type, count - 1

        # unique check
        descriptive_id = new_ind.graph.descriptive_id
        if descriptive_id in pop_graph_descriptive_ids:
            # worker can't send nonempty string! wtf?
            return '', individual, mutation_type, count - 1
        pop_graph_descriptive_ids[descriptive_id] = True

        # evaluation
        new_inds = evaluator([new_ind])
        if not new_inds:
            # worker can't send nonempty string! wtf?
            return '', individual, mutation_type, count - 1

        return None, new_inds[0], mutation_type, count - 1

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
