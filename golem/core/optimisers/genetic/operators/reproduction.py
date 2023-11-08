import time
from concurrent.futures import as_completed
from copy import deepcopy, copy
from dataclasses import dataclass
from enum import Enum
from itertools import cycle, chain
from multiprocessing.managers import DictProxy
from multiprocessing import Manager
from random import choice, sample
from typing import Optional, Dict, Union, List

from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.backend.queues import Queue

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS_PER_IND
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover, CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation, SinglePredefinedMutation, MutationType
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
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
        new_population = self._reproduce(new_population, evaluator)
        self._check_final_population(new_population)
        return new_population

    def _reproduce(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """Generate new individuals by mutation in parallel.
           Implements additional checks on population to ensure that population size is greater or equal to
           required population size. Also controls uniqueness of population.
        """
        with Manager() as manager:
            left_tries = self.parameters.pop_size * MAX_GRAPH_GEN_ATTEMPTS_PER_IND
            new_population = list()
            pop_graph_descriptive_ids = manager.dict({ids: True for ids in self._pop_graph_descriptive_ids})
            task_queue, result_queue, experience_queue = [manager.Queue() for _ in range(3)]

            worker = ReproduceWorker(crossover=self.crossover, mutation=self.mutation,
                                     verifier=self.verifier, evaluator=evaluator,
                                     pop_graph_descriptive_ids=pop_graph_descriptive_ids,
                                     population=population,
                                     task_queue=task_queue, result_queue=result_queue,
                                     experience_queue=experience_queue)

            # TODO there is problem with random seed in parallel workers
            # TODO tries in one thread

            # create pool with workers
            executor = get_reusable_executor(max_workers=self.mutation.requirements.n_jobs)
            for _ in range(max(1, self.mutation.requirements.n_jobs - 1)):
                executor.submit(worker)

            try:
                # create new population
                while left_tries > 0 and len(new_population) < self.parameters.pop_size:
                    # if there is not enough jobs, create new empty job
                    # for fully random starting individuals and operation types
                    while task_queue.qsize() < 2:
                        tries = self.parameters.max_num_of_reproducer_attempts
                        task_queue.put(ReproducerWorkerTask(crossover_tries=tries, mutation_tries=tries))
                        time.sleep(0.01)  # get workers some time to get tasks from queue

                    # process result
                    if result_queue.qsize() > 0:
                        failed_stage, individual, mutation_type = result_queue.get()
                        left_tries -= 1
                        new_population.append(individual)

                    # process unsuccessful creation attempt
                    if experience_queue.qsize() > 0:
                        failed_stage, individual_uid, mutation_type = experience_queue.get()
                        left_tries -= 1
                        if failed_stage is ReproducerWorkerStageEnum.MUTATION_VERIFICATION:
                            # experience for mab
                            self.mutation.agent_experience.collect_experience(population_uid_map[individual_uid],
                                                                              mutation_type,
                                                                              reward=-1.0)
            finally:
                # shutdown workers
                executor.shutdown(wait=False)

            # update looked graphs
            self._pop_graph_descriptive_ids |= set(pop_graph_descriptive_ids.keys())

            # rebuild population
            new_population = self._rebuild_final_population(population=population, new_population=new_population)
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

    def _rebuild_final_population(self, population: PopulationT, new_population: PopulationT) -> PopulationT:
        """ Recreate new_population in main thread with parents from population """
        population_uid_map = {individual.uid: individual for individual in population}
        rebuilded_population = []
        for individual in new_population:
            if individual.parent_operator:
                parent_uid = individual.parent_operator.parent_individuals[0].uid
                parent_operator = ParentOperator(type_=individual.parent_operator.type_,
                                                 operators=individual.parent_operator.operators,
                                                 parent_individuals=population_uid_map[parent_uid])
            else:
                parent_operator = None
            individual = Individual(deepcopy(individual.graph),
                                    parent_operator,
                                    fitness=individual.fitness,
                                    metadata=self.mutation.requirements.static_individual_metadata)
            rebuilded_population.append(individual)
        return rebuilded_population


class ReproducerWorkerStageEnum(Enum):
    # TODO test that check that nums start from 0 and go to max with 1 steps
    FINISH = -100
    CROSSOVER = 0
    CROSSOVER_VERIFICATION = 1
    CROSSOVER_UNIQUENESS_CHECK = 2
    CROSSOVER_EVALUATION = 3
    MUTATION = 4
    MUTATION_VERIFICATION = 5
    MUTATION_UNIQUENESS_CHECK = 6
    MUTATION_EVALUATION = 7


@dataclass
class ReproducerWorkerTask:
    stage: ReproducerWorkerStageEnum = ReproducerWorkerStageEnum(0)
    fail: bool = False

    # crossover data
    graph_1_for_crossover: Optional[OptGraph] = None
    graph_2_for_crossover: Optional[OptGraph] = None
    crossover_type: Optional[CrossoverTypesEnum] = None
    crossover_tries: int = 1
    crossover_fitness = Optional[Fitness] = None

    # mutation data
    graph_for_mutation: Optional[OptGraph] = None
    mutation_type: Optional[MutationType] = None
    mutation_tries: int = 1

    # result
    final_graph: Optional[OptGraph] = None
    final_fitness = Optional[Fitness] = None

    @property
    def is_crossover(self):
        return self.stage in [ReproducerWorkerStageEnum.CROSSOVER,
                              ReproducerWorkerStageEnum.CROSSOVER_VERIFICATION,
                              ReproducerWorkerStageEnum.CROSSOVER_UNIQUENESS_CHECK,
                              ReproducerWorkerStageEnum.CROSSOVER_EVALUATION]

    @property
    def is_mutation(self):
        return not self.is_crossover

    @property
    def tries(self):
        return self.crossover_tries if self.is_crossover else self.mutation_tries

    @property.setter
    def tries(self, value: int):
        if self.is_crossover:
            self.crossover_tries = value
        else:
            self.mutation_tries = value

    def step_in_stage(self, flag: bool):
        step = 1 if flag else -1
        self.stage = ReproducerWorkerStageEnum(self.stage.value + step)


class ReproduceWorker:
    def __init__(self,
                 crossover: Crossover,
                 mutation: MutationType,
                 verifier: GraphVerifier,
                 evaluator: EvaluationOperator,
                 pop_graph_descriptive_ids: Union[DictProxy, Dict],
                 population: PopulationT,
                 task_queue: Queue,
                 result_queue: Queue,
                 experience_queue: Queue
                 ):
        self.crossover = crossover
        self.mutation = mutation
        self.verifier = verifier
        self.evaluator = evaluator
        self._pop_graph_descriptive_ids = pop_graph_descriptive_ids
        self._population = population
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._experience_queue = experience_queue

    def __call__(self):
        tasks = []
        while True:
            # work with existing task from tasks or from queue
            if not tasks:
                tasks.append(self._task_queue.get())
            processed_tasks = self.process_task(tasks.pop())

            # process result
            for processed_task in processed_tasks:
                if processed_task.stage is ReproducerWorkerStageEnum.FINISH:
                    self._result_queue.put(processed_task)
                    continue
                if processed_task.fail:
                    self._experience_queue.put(processed_task)
                if processed_task.tries > 0:
                    # task is not finished, need new try
                    tasks.append(processed_task)

            # if there are some tasks, add it to parallel queue
            for _ in range(len(tasks)):
                self._task_queue.put(tasks.pop())

    def process_task(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        task = copy(task)  # input task

        # crossover
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER:
            return self.crossover_stage(task)

        # crossover result verification
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER_VERIFICATION:
            task.step_in_stage(self.verifier(task.graph_for_mutation))
            return [task]

        # crossover uniqueness check
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER_UNIQUENESS_CHECK:
            return self.uniqueness_check_stage(task)

        # crossover result evaluation
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER_EVALUATION:
            return self.evaluation_stage(task)

        # mutation
        # TODO add some mutation for each crossover result
        if task.stage is ReproducerWorkerStageEnum.MUTATION:
            return self.mutation_stage(task)

        # mutation result verification
        if task.stage is ReproducerWorkerStageEnum.MUTATION_VERIFICATION:
            task.step_in_stage(self.verifier(task.final_graph))
            return [task]

        # mutation uniqueness check
        if task.stage is ReproducerWorkerStageEnum.MUTATION_UNIQUENESS_CHECK:
            return self.uniqueness_check_stage(task)

        # mutation result evaluation
        if task.stage is ReproducerWorkerStageEnum.MUTATION_EVALUATION:
            return self.evaluation_stage(task)

    def crossover_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        tasks = []  # tasks to return

        # if there is no graphs for crossover then get random graphs
        if task.graph_1_for_crossover is None or task.graph_1_for_crossover is None:
            inds_for_crossover = sample(self._population, k=2)
            task.graph_1_for_crossover, task.graph_2_for_crossover = [ind.graph for ind in inds_for_crossover]

        # make crossover
        task.crossover_tries -= 1
        *new_graphs, task.crossover_type = self.crossover(task.graph_1_for_crossover,
                                                          task.graph_2_for_crossover,
                                                          task.crossover_type)

        # if there is no new_graphs than go to new try
        if not new_graphs:
            tasks.append(task)
        else:
            # create new task for each new graph after crossover for next stage
            task.step_in_stage(True)
            for graph in new_graphs:
                new_task = copy(task)
                new_task.graph_for_mutation = graph
                tasks.append(new_task)
        return tasks

    def mutation_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        task.final_graph, task.mutation_type = self.mutation(task.graph_for_mutation, task.mutation_type)
        task.mutation_tries -= 1
        task.step_in_stage(True)
        return [task]

    def uniqueness_check_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        graph = task.graph_for_mutation if task.is_crossover else task.final_graph
        descriptive_id = graph.descriptive_id
        if descriptive_id not in self._pop_graph_descriptive_ids:
            self._pop_graph_descriptive_ids[descriptive_id] = True
            task.step_in_stage(True)
        else:
            task.step_in_stage(False)
        return [task]

    def evaluation_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        graph = task.graph_for_mutation if task.is_crossover else task.final_graph
        individual = Individual(deepcopy(graph), metadata=self.mutation.requirements.static_individual_metadata)
        evaluated_individuals = self.evaluator([individual])
        if evaluated_individuals:
            # TODO add null_fitness as flag for previous stage
            task.step_in_stage(True)
        else:
            task.step_in_stage(False)
        # TODO return fitness
        return [task]


