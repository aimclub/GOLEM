import time
from copy import deepcopy, copy
from dataclasses import dataclass
from enum import Enum
from multiprocessing.managers import DictProxy
from multiprocessing import Manager
from queue import Empty, Queue
from random import sample, randint
from typing import Optional, Dict, Union, List

from joblib import Parallel, delayed

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS_PER_IND
from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover, CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationType
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.utilities.random import RandomStateHandler


class ReproducerWorkerStageEnum(Enum):
    # TODO test that check that nums start from 0 and go to max (FINISH) with 1 steps
    (CROSSOVER, CROSSOVER_VERIFICATION, CROSSOVER_UNIQUENESS_CHECK, CROSSOVER_EVALUATION,
     MUTATION, MUTATION_VERIFICATION, MUTATION_UNIQUENESS_CHECK, MUTATION_EVALUATION, FINISH) = range(9)

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


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
        new_population = self._reproduce(selected_individuals, evaluator)
        self._check_final_population(new_population)
        return new_population

    def _reproduce(self, population: PopulationT, evaluator: EvaluationOperator) -> PopulationT:
        """Generate new individuals by mutation in parallel.
           Implements additional checks on population to ensure that population size is greater or equal to
           required population size. Also controls uniqueness of population.
        """
        with Manager() as manager:
            pop_graph_descriptive_ids = manager.dict({ids: True for ids in self._pop_graph_descriptive_ids})
            task_queue, result_queue, failed_queue = [manager.Queue() for _ in range(3)]

            # empty task for worker if there is no work
            empty_task = ReproducerWorkerTask(
                crossover_tries=self.parameters.max_num_of_crossover_reproducer_attempts,
                mutation_tries=self.parameters.max_num_of_mutation_reproducer_attempts,
                mutation_attempts_per_each_crossover=self.parameters.mutation_attempts_per_each_crossover_reproducer)

            # parameters for worker
            worker_parameters = dict(crossover=self.crossover, mutation=self.mutation,
                                     verifier=self.verifier, evaluator=evaluator,
                                     pop_graph_descriptive_ids=pop_graph_descriptive_ids,
                                     population=population,
                                     task_queue=task_queue, result_queue=result_queue,
                                     failed_queue=failed_queue, empty_task=empty_task,
                                     log=self._log)

            n_jobs = self.mutation.requirements.n_jobs
            with Parallel(n_jobs=n_jobs + 1, prefer='processes', return_as='generator') as parallel:
                # prepare (n_jobs + 1) workers
                workers = [ReproduceWorker(seed=randint(0, int(2**32 - 1)), **worker_parameters)
                           for _ in range(n_jobs + 1)]
                # run n_jobs workers with run_flag = True
                # and one worker with run_flag = False
                # It guarantees n_jobs workers parallel execution also if n_jobs == 1
                # because joblib for n_jobs == 1 does not start parallel pool
                _ = parallel(delayed(worker)(run_flag) for worker, run_flag in zip(workers, [True] * n_jobs + [False]))

                finished_tasks, failed_tasks = list(), list()
                left_tries = self.parameters.pop_size * MAX_GRAPH_GEN_ATTEMPTS_PER_IND * n_jobs
                while left_tries > 0 and len(finished_tasks) < self.parameters.pop_size:
                    # main thread is fast
                    # frequent queues blocking with qsize is not good idea
                    time.sleep(1)
                    while failed_queue.qsize() > 0:
                        left_tries -= 1
                        failed_tasks.append(failed_queue.get())

                    while result_queue.qsize() > 0:
                        left_tries -= 1
                        finished_tasks.append(result_queue.get())

            # get all finished works
            while failed_queue.qsize() > 0:
                failed_tasks.append(failed_queue.get())
            while result_queue.qsize() > 0:
                finished_tasks.append(result_queue.get())

            # update looked graphs
            self._pop_graph_descriptive_ids |= set(pop_graph_descriptive_ids.keys())

        # rebuild population
        new_population = self._process_tasks(population=population,
                                             finished_tasks=finished_tasks,
                                             failed_tasks=failed_tasks)
        return new_population

    def _process_tasks(self,
                       population: PopulationT,
                       finished_tasks: List['ReproducerWorkerTask'],
                       failed_tasks: List['ReproducerWorkerTask']):
        population_uid_map = {ind.uid: ind for ind in population}

        crossover_individuals, new_population = dict(), []
        for task in finished_tasks + failed_tasks:
            if task.stage > ReproducerWorkerStageEnum.MUTATION:
                uids = (task.graph_1_uid, task.graph_2_uid)
                # create individuals, generated by crossover
                if uids not in crossover_individuals:
                    individuals = self.crossover._get_individuals(new_graphs=[task.graph_for_mutation],
                                                                  parent_individuals=[population_uid_map[uid]
                                                                                      for uid in uids],
                                                                  crossover_type=task.crossover_type,
                                                                  fitness=task.crossover_fitness)
                    crossover_individuals[uids] = individuals[0]

                # create individuals, generated by mutation
                if uids in crossover_individuals:
                    individual = self.mutation._get_individual(new_graph=task.final_graph,
                                                               mutation_type=task.mutation_type,
                                                               parent=crossover_individuals[uids],
                                                               fitness=task.final_fitness)
                    if task.stage is ReproducerWorkerStageEnum.FINISH:
                        new_population.append(individual)
                    elif task.failed_stage is ReproducerWorkerStageEnum.MUTATION_VERIFICATION:
                        # experience for mab
                        self.mutation.agent_experience.collect_experience(individual, task.mutation_type, reward=-1.0)
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


@dataclass
class ReproducerWorkerTask:
    stage: ReproducerWorkerStageEnum = ReproducerWorkerStageEnum(0)
    _fail: bool = False
    failed_stage: ReproducerWorkerStageEnum = None
    mutation_attempts_per_each_crossover: int = 1

    # crossover data
    graph_1_uid: Optional[str] = None
    graph_2_uid: Optional[str] = None
    graph_1_for_crossover: Optional[OptGraph] = None
    graph_2_for_crossover: Optional[OptGraph] = None
    crossover_type: Optional[CrossoverTypesEnum] = None
    crossover_tries: int = 1
    crossover_fitness: Optional[Fitness] = None

    # mutation data
    graph_for_mutation: Optional[OptGraph] = None
    mutation_type: Optional[MutationType] = None
    mutation_tries: int = 1

    # result
    final_graph: Optional[OptGraph] = None
    final_fitness: Optional[Fitness] = None

    @property
    def fail(self):
        return self._fail

    @fail.setter
    def fail(self, value):
        if value:
            self.failed_stage = self.stage
        self._fail = value

    @property
    def is_crossover(self):
        return self.stage < ReproducerWorkerStageEnum.MUTATION

    @property
    def tries(self):
        return self.crossover_tries if self.is_crossover else self.mutation_tries

    def step_in_stage(self, steps: int):
        self.stage = ReproducerWorkerStageEnum(self.stage.value + steps)


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
                 failed_queue: Queue,
                 empty_task: ReproducerWorkerTask,
                 seed: int,
                 log
                 ):
        self.crossover = crossover
        self.mutation = mutation
        self.verifier = verifier
        self.evaluator = evaluator
        self._pop_graph_descriptive_ids = pop_graph_descriptive_ids
        self._population = population
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._failed_queue = failed_queue
        self._empty_task = empty_task
        self._seed = seed
        self._log = log

    def __call__(self, run: bool = True):
        self._log.warning(f"CALLED")
        with RandomStateHandler(self._seed):
            tasks = [self._empty_task]
            self._log.warning(f"START CYCLE")
            while run:
                # is there is no tasks, try to get 1. task from queue 2. empty task
                if not tasks:
                    try:
                        tasks.append(self._task_queue.get(timeout=0.02))
                    except Empty:
                        tasks.append(self._empty_task)

                # work with task
                processed_tasks = self.process_task(tasks.pop())

                # process result
                for processed_task in processed_tasks:
                    if processed_task.stage is ReproducerWorkerStageEnum.FINISH:
                        self._result_queue.put(processed_task)
                        continue
                    if processed_task.fail:
                        self._log.warning(f"FAIL: {processed_task.failed_stage}")
                        self._failed_queue.put(processed_task)
                        if processed_task.tries > 0:
                            tasks.append(processed_task)
                    else:
                        tasks.append(processed_task)

                # if there are some tasks, add it to parallel queue
                for _ in range(len(tasks) - 1):
                    self._task_queue.put(tasks.pop())

    def process_task(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        """ Get task, make 1 stage and return processed task """
        # self._log.warning(f"START: {task.stage} {task.crossover_tries}:{task.mutation_tries}")
        task = copy(task)  # input task
        task.fail = False

        # crossover
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER:
            return self.crossover_stage(task)

        # crossover result verification
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER_VERIFICATION:
            task.fail = not self.verifier(task.graph_for_mutation)
            task.step_in_stage(-1 if task.fail else 1)
            return [task]

        # crossover uniqueness check
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER_UNIQUENESS_CHECK:
            task.step_in_stage(1)
            return [task]
            # processed_task = self.uniqueness_check_stage(task)[0]
            # processed_task.step_in_stage(-2 if processed_task.fail else 1)
            # return [processed_task]

        # crossover result evaluation
        if task.stage is ReproducerWorkerStageEnum.CROSSOVER_EVALUATION:
            task.step_in_stage(1)
            return [copy(task) for _ in range(task.mutation_attempts_per_each_crossover)]
        #     processed_task = self.evaluation_stage(task)[0]
        #     if processed_task.fail:
        #         processed_task.step_in_stage(-3)
        #         return [processed_task]
        #     else:
        #         # create some tasks for mutation for crossover result
        #         processed_task.step_in_stage(1)
        #         return [copy(processed_task) for _ in range(task.mutation_attempts_per_each_crossover)]

        # mutation
        if task.stage is ReproducerWorkerStageEnum.MUTATION:
            return self.mutation_stage(task)

        # mutation result verification
        if task.stage is ReproducerWorkerStageEnum.MUTATION_VERIFICATION:
            task.fail = not self.verifier(task.final_graph)
            task.step_in_stage(-1 if task.fail else 1)
            return [task]

        # mutation uniqueness check
        if task.stage is ReproducerWorkerStageEnum.MUTATION_UNIQUENESS_CHECK:
            processed_task = self.uniqueness_check_stage(task)[0]
            processed_task.step_in_stage(-2 if processed_task.fail else 1)
            return [processed_task]

        # mutation result evaluation
        if task.stage is ReproducerWorkerStageEnum.MUTATION_EVALUATION:
            processed_task = self.evaluation_stage(task)[0]
            processed_task.step_in_stage(-3 if processed_task.fail else 1)
            return [processed_task]

    def crossover_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        tasks = []  # tasks to return

        # if there is no graphs for crossover then get random graphs
        if task.graph_1_for_crossover is None or task.graph_1_for_crossover is None:
            inds_for_crossover = sample(self._population, k=2)
            task.graph_1_uid, task.graph_1_for_crossover = inds_for_crossover[0].uid, inds_for_crossover[0].graph
            task.graph_2_uid, task.graph_2_for_crossover = inds_for_crossover[1].uid, inds_for_crossover[1].graph

        # make crossover
        task.crossover_tries -= 1
        *new_graphs, task.crossover_type = self.crossover(task.graph_1_for_crossover,
                                                          task.graph_2_for_crossover,
                                                          task.crossover_type)

        if not new_graphs:
            # if there is no new_graphs then go to new try
            task.fail = True
            tasks.append(task)
        else:
            # create new task for each new graph after crossover for next stage
            task.step_in_stage(1)
            for graph in new_graphs:
                new_task = copy(task)
                new_task.graph_for_mutation = graph
                tasks.append(new_task)
        return tasks

    def mutation_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        task.final_graph, task.mutation_type = self.mutation(task.graph_for_mutation, task.mutation_type)
        task.mutation_tries -= 1
        if task.final_graph is None:
            task.fail = True
        else:
            task.step_in_stage(1)
        return [task]

    def uniqueness_check_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        if task.is_crossover:
            graph = task.graph_for_mutation
        else:
            graph = task.final_graph
        descriptive_id = graph.descriptive_id
        if descriptive_id not in self._pop_graph_descriptive_ids:
            self._pop_graph_descriptive_ids[descriptive_id] = True
            task.fail = False
        else:
            task.fail = True
        return [task]

    def evaluation_stage(self, task: ReproducerWorkerTask) -> List[ReproducerWorkerTask]:
        if task.is_crossover:
            graph = task.graph_for_mutation
        else:
            graph = task.final_graph
        individual = Individual(deepcopy(graph), metadata=self.mutation.requirements.static_individual_metadata)
        evaluated_individuals = self.evaluator([individual])
        if evaluated_individuals:# and evaluated_individuals[0].fitness.valid:
            task.fail = False
            if task.is_crossover:
                task.crossover_fitness = evaluated_individuals[0].fitness
            else:
                task.final_fitness = evaluated_individuals[0].fitness
        else:
            task.fail = True
        return [task]
