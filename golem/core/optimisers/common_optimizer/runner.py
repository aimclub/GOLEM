from abc import abstractmethod
from collections import deque

from typing import List, Callable

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum


class Runner:
    def __init__(self):
        pass

    @abstractmethod
    def run(self, scheme: Scheme, task: Task, nodes: List[Node], stop_fun: Callable):
        raise NotImplementedError('It is abstract method')


class ParallelRunner(Runner):
    def __init__(self):
        super().__init__()


class OneThreadRunner(Runner):
    def __init__(self):
        super().__init__()

    def run(self, scheme: Scheme, task: Task, nodes: List[Node], stop_fun: Callable):
        origin_task = task
        node_map = {node.name: node for node in nodes}
        queued_tasks, finished_tasks, all_tasks = [list() for i in range(3)]
        while not stop_fun(finished_tasks, all_tasks):
            task = origin_task.copy() if len(queued_tasks) == 0 else queued_tasks.pop()
            all_tasks.append(task)
            task = scheme.next(task)
            if task.status is TaskStatusEnum.FINISH:
                finished_tasks.append(task)
                continue
            queued_tasks.extend(node_map[task.node](task))
        return finished_tasks
