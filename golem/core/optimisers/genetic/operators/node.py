from dataclasses import dataclass, replace
from queue import Queue
from typing import Optional, List, Union, Any

from golem.core.optimisers.genetic.operators.operator import Operator
from golem.core.optimisers.graph import OptGraph


@dataclass
class GeneticOperatorTask:
    stage: 'GeneticNode'

    graphs: List[OptGraph]
    operator_type: Optional[Any] = None

    # parent data
    parent_task: Optional['GeneticOperatorTask'] = None

    fail: bool = False
    fail_message: str = ''
    left_tries: int = 1

    def copy(self):
        return replace(self)

    def create_failed_task(self, exception: Exception):
        failed_task = self.copy()
        failed_task.fail = True
        failed_task.fail_message = exception.__str__()
        failed_task.left_tries -= 1
        return failed_task

    def create_successive_task(self, graphs: List[OptGraph], operator_type: Any):
        task = self.copy()
        task.graphs = graphs
        task.operator_type = operator_type
        task.parent_task = self
        return task


@dataclass(frozen=True)
class GeneticNode:
    name: str
    operator: Operator
    processed_queue: Queue
    success_outputs: List['GeneticNode'] = None
    fail_outputs: Optional[List['GeneticNode']] = None

    def __call__(self, task: GeneticOperatorTask):
        if task.left_tries > 0:
            try:
                *grouped_graphs, operator_type = self.operator(task.graphs, task.operator_type)
                tasks = [task.create_successive_task(graphs, operator_type) for graphs in grouped_graphs]
                next_nodes = self.success_outputs
            except Exception as exception:
                tasks = [task.create_failed_task(exception)]
                next_nodes = self.fail_outputs

            final_tasks = list()
            for _task in tasks:
                for _node in next_nodes:
                    new_task = _task.copy()
                    new_task.stage = _node
                    final_tasks.append(new_task)
            return final_tasks
