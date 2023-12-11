import time
from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from multiprocessing import Manager
from queue import Empty
from random import randint

from typing import List, Callable, Dict

from joblib import Parallel, delayed

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum
from golem.utilities.random import RandomStateHandler
from golem.utilities.utilities import determine_n_jobs


@dataclass
class Worker:
    """
    Represents a worker that executes tasks within the optimization process.

    Args:
        scheme: optimization scheme of nodes execution.
        origin_task: task to start the execution from.
        node_map: mapping of node names to Node objects.
        queued_tasks: queue to store the queued tasks.
        processed_tasks: queue to store the processed tasks.
        sleep_time: sleep time in seconds between each task execution.
    """

    scheme: Scheme
    origin_task: Task
    node_map: Dict[str, Node]
    queued_tasks: 'Queue'
    processed_tasks: 'Queue'
    sleep_time: float

    def __call__(self, seed: int):
        with RandomStateHandler(seed):
            task = self.origin_task.copy()
            while True:
                task = self.scheme.next(task)
                if task.status is TaskStatusEnum.FINISH:
                    self.processed_tasks.put(task)
                    try:
                        task = self.queued_tasks.get(timeout=self.sleep_time)
                    except Empty:
                        task = self.origin_task.copy()
                else:
                    new_tasks = self.node_map[task.node](task)
                    for task in new_tasks[:-1]:
                        self.queued_tasks.put(task)
                    task = new_tasks[-1]


class Runner:
    """
    Abstract base class for runners in the optimization process.
    """
    @abstractmethod
    def run(self, scheme: Scheme, task: Task, nodes: List[Node], stop_fun: Callable):
        raise NotImplementedError('It is abstract method')


class ParallelRunner(Runner):
    """
    Runner that executes tasks in parallel using multiple processes.
    Args:
        n_jobs: number of processes to use for parallel execution. Defaults to -1.
        main_cycle_sleep_seconds: sleep time in seconds between each main cycle. Defaults to 1.
        worker_cycle_sleep_seconds: sleep time in seconds between each worker cycle. Defaults to 0.02.
    """

    # TODO test for same results from Parallel and OneThread
    def __init__(self,
                 *args,
                 n_jobs: int = -1,
                 main_cycle_sleep_seconds: float = 1,
                 worker_cycle_sleep_seconds: float = 0.02,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.n_jobs = n_jobs
        self.main_cycle_sleep_seconds = main_cycle_sleep_seconds
        self.worker_cycle_sleep_seconds = worker_cycle_sleep_seconds

    def run(self, scheme: Scheme, task: Task, nodes: List[Node], stop_fun: Callable):
        with Manager() as m:
            queued_tasks, processed_tasks = m.Queue(), m.Queue()
            worker = Worker(origin_task=task, scheme=scheme, node_map={node.name: node for node in nodes},
                            queued_tasks=queued_tasks, processed_tasks=processed_tasks,
                            sleep_time=self.worker_cycle_sleep_seconds)
            with Parallel(n_jobs=self.n_jobs, prefer='processes', return_as='generator') as parallel:
                _ = parallel(delayed(worker)(randint(0, int(2 ** 32 - 1))) for _ in range(self.n_jobs))
                finished_tasks, all_tasks = list(), list()
                while not stop_fun(finished_tasks, all_tasks):
                    time.sleep(self.main_cycle_sleep_seconds)
                    for _ in range(processed_tasks.qsize()):
                        task = processed_tasks.get()
                        all_tasks.append(task)
                        if task.status is TaskStatusEnum.FINISH:
                            finished_tasks.append(task)
        return finished_tasks


class OneThreadRunner(Runner):
    """
    Runner that executes tasks in a single thread.
    """

    def run(self, scheme: Scheme, task: Task, nodes: List[Node], stop_fun: Callable):
        origin_task = task
        node_map = {node.name: node for node in nodes}
        queued_tasks, finished_tasks, all_tasks = [list() for i in range(3)]
        while not stop_fun(finished_tasks, all_tasks):
            task = origin_task.copy() if len(queued_tasks) == 0 else queued_tasks.pop()
            task = scheme.next(task)
            if task.status is TaskStatusEnum.FINISH:
                finished_tasks.append(task)
                tasks = [task]
            else:
                tasks = node_map[task.node](task)
                queued_tasks.extend(tasks)
            all_tasks.extend(tasks)
        return finished_tasks
