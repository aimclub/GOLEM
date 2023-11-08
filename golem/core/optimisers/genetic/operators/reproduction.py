import time
from concurrent.futures import as_completed
from copy import deepcopy
from enum import Enum
from itertools import cycle, chain
from multiprocessing.managers import DictProxy
from multiprocessing import Manager
from typing import Optional, Dict, Union

from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.backend.queues import Queue

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS_PER_IND
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation, SinglePredefinedMutation, MutationType
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.core.optimisers.opt_history_objects.individual import Individual


class FailedStageEnum(Enum):
    NONE = 0
    MUTATION = 1
    VERIFICATION = 2
    UNIQUENESS_CHECK = 3
    EVALUATION = 4


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
                 verifier: Optional[GraphVerifier] = None):
        self.parameters = parameters
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover
        self.verifier = verifier or self.mutation.graph_generation_params.verifier

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
        if self.mutation.requirements.n_jobs > 1:
            new_population = self._mutate_over_population_parallel(population, evaluator)
        else:
            new_population = self._mutate_over_population_single_thread(population, evaluator)
        return new_population

    def _mutate_over_population_single_thread(self,
                                              population: PopulationT,
                                              evaluator: EvaluationOperator) -> PopulationT:
        left_tries = self.parameters.pop_size * MAX_GRAPH_GEN_ATTEMPTS_PER_IND
        new_population, cycled_population = [], cycle(population)
        mutation = SinglePredefinedMutation(parameters=self.mutation.parameters,
                                            requirements=self.mutation.requirements,
                                            graph_gen_params=self.mutation.graph_generation_params,
                                            mutations_repo=self.mutation._mutations_repo)
        pop_graph_descriptive_ids = {key: True for key in self._pop_graph_descriptive_ids}

        while left_tries > 0 and len(new_population) < self.parameters.pop_size:
            individual = next(cycled_population)
            mutation_type = self.mutation.agent.choose_action(individual.graph)

            failed_stage, individual, mutation_type, retained_tries = \
                self._mutation_n_evaluation(individual=next(cycled_population),
                                            tries=1,
                                            mutation_type=mutation_type,
                                            pop_graph_descriptive_ids=pop_graph_descriptive_ids,
                                            mutation=mutation,
                                            evaluator=evaluator)
            left_tries -= 1
            if failed_stage is FailedStageEnum.NONE:
                new_population.append(individual)

        # update looked graphs
        self._pop_graph_descriptive_ids |= set(pop_graph_descriptive_ids.keys())
        return new_population

    def _mutate_over_population_parallel(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """Generate new individuals by mutation in parallel.
           Implements additional checks on population to ensure that population size is greater or equal to
           required population size. Also controls uniqueness of population.
        """
        with Manager() as manager:
            population_uid_map = {ind.uid: ind for ind in population}
            left_tries = self.parameters.pop_size * MAX_GRAPH_GEN_ATTEMPTS_PER_IND
            cycled_population_uid = cycle(population_uid_map)
            new_population, inds_for_experience = list(), list()
            mutation = SinglePredefinedMutation(parameters=self.mutation.parameters,
                                                requirements=self.mutation.requirements,
                                                graph_gen_params=self.mutation.graph_generation_params,
                                                mutations_repo=self.mutation._mutations_repo)
            pop_graph_descriptive_ids = manager.dict({ids: True for ids in self._pop_graph_descriptive_ids})
            task_queue, result_queue = manager.Queue(), manager.Queue()

            def worker(pop_graph_descriptive_ids: DictProxy = pop_graph_descriptive_ids,
                       mutation: SinglePredefinedMutation = mutation,
                       evaluator: EvaluationOperator = evaluator,
                       population_uid_map: Dict[str, Individual] = population_uid_map,
                       task_queue: Queue = task_queue,
                       result_queue: Queue = result_queue) -> None:
                while True:
                    individual_uid, mutation_type, tries = task_queue.get()

                    failed_stage, individual, mutation_type, retained_tries = \
                        self._mutation_n_evaluation(individual=population_uid_map[individual_uid],
                                                    tries=tries,
                                                    mutation_type=mutation_type,
                                                    pop_graph_descriptive_ids=pop_graph_descriptive_ids,
                                                    mutation=mutation,
                                                    evaluator=evaluator)

                    if failed_stage is FailedStageEnum.NONE:
                        result_queue.put((failed_stage, individual, mutation_type))
                    else:
                        result_queue.put((failed_stage, individual_uid, mutation_type))
                        task_queue.put((individual_uid, mutation_type, tries - 1))


            # create pool
            executor = get_reusable_executor(max_workers=self.mutation.requirements.n_jobs)
            for _ in range(max(2, self.mutation.requirements.n_jobs - 1)): executor.submit(worker)

            while left_tries > 0 and len(new_population) < self.parameters.pop_size:
                # if there is not enough jobs, create new
                while task_queue.qsize() < 2:
                    individual_uid = next(cycled_population_uid)
                    mutation_type = self.mutation.agent.choose_action(population_uid_map[individual_uid].graph)
                    task_queue.put((individual_uid, mutation_type, self.parameters.max_num_of_mutation_attempts))
                    time.sleep(0.01)

                # process result
                if result_queue.qsize() > 0:
                    failed_stage, individual, mutation_type = result_queue.get()
                    left_tries -= 1
                    if failed_stage is FailedStageEnum.NONE:
                        new_population.append(individual)
                    elif failed_stage is FailedStageEnum.VERIFICATION:
                        inds_for_experience.append((population_uid_map[individual], mutation_type))

            # shutdown workers
            executor.shutdown(wait=False)

            # add experience for agent
            for individual, mutation_type in inds_for_experience:
                self.mutation.agent_experience.collect_experience(individual, mutation_type, reward=-1.0)

            # update looked graphs
            self._pop_graph_descriptive_ids |= set(pop_graph_descriptive_ids.keys())

            # rebuild population due to problem with changing id of individuals in parallel individuals building
            rebuilded_population = []
            for individual in new_population:
                if individual.parent_operator:
                    parent_operator = ParentOperator(type_=individual.parent_operator.type_,
                                                     operators=individual.parent_operator.operators,
                                                     parent_individuals=population_uid_map[
                                                         individual.parent_operator.parent_individuals[0].uid])
                else:
                    parent_operator = None
                individual = Individual(deepcopy(individual.graph),
                                        parent_operator,
                                        fitness=individual.fitness,
                                        metadata=self.mutation.requirements.static_individual_metadata)
                rebuilded_population.append(individual)
            return rebuilded_population

    def _mutation_n_evaluation(self,
                               individual: Individual,
                               tries: int,
                               mutation_type: MutationType,
                               pop_graph_descriptive_ids: Union[Dict, DictProxy],
                               mutation: SinglePredefinedMutation,
                               evaluator: EvaluationOperator):
        # mutation
        new_ind = mutation(individual, mutation_type=mutation_type)
        if not new_ind:
            return FailedStageEnum.MUTATION, individual, mutation_type, tries - 1

        # verification
        if not self.verifier(new_ind.graph):
            return FailedStageEnum.VERIFICATION, individual, mutation_type, tries - 1

        # unique check
        descriptive_id = new_ind.graph.descriptive_id
        if descriptive_id in pop_graph_descriptive_ids:
            return FailedStageEnum.UNIQUENESS_CHECK, individual, mutation_type, tries - 1
        pop_graph_descriptive_ids[descriptive_id] = True

        # evaluation
        new_inds = evaluator([new_ind])
        if not new_inds:
            return FailedStageEnum.EVALUATION, individual, mutation_type, tries - 1

        return FailedStageEnum.NONE, new_inds[0], mutation_type, tries - 1

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