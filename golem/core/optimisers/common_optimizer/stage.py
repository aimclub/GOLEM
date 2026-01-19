from dataclasses import dataclass

from typing import List, Callable, Type

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.runner import Runner
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.task import Task


@dataclass
class Stage:
    """
    Represents a stage in runner and stores operations (nodes) and the operations execution scheme.
        task_builder - Task class
        stop_fun - see Runner docs
        parameter_updater - callable that gets results of Runner call and parameters, returns updated parameters
    Args:
        runner: object responsible for executing tasks.
        nodes: list of operations in the stage.
        task_builder: Task class used to create tasks from parameters.
        scheme: object defining the data flow between nodes.
        stop_fun: stop function used by the runner.
        parameter_updater: function that gets updated parameters from runner call and passed parameters.
    """
    # TODO move to Runner

    runner: Runner
    nodes: List[Node]
    task_builder: Type[Task]
    scheme: Scheme
    stop_fun: Callable[[List[Task], List[Task]], bool]
    parameter_updater: Callable[[List[Task], 'CommonOptimizerParameters'], 'CommonOptimizerParameters']

    def run(self, parameters: 'CommonOptimizerParameters'):
        """  Executes the stage's runner with the specified parameters and returns the updated parameters. """
        task = self.task_builder(parameters)
        results = self.runner.run(nodes=self.nodes, task=task, scheme=self.scheme, stop_fun=self.stop_fun)
        return self.parameter_updater(results, parameters)
