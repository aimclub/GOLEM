from dataclasses import dataclass

from typing import List, Callable

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.runner import Runner
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.task import Task


@dataclass
class Stage:
    runner: Runner
    nodes: List[Node]
    task_builder: Callable[['CommonOptimizerParameters'], Task]
    scheme: Scheme
    stop_fun: Callable[[List[Task], List[Task]], bool]
    parameter_updater: Callable[[List[Task], 'CommonOptimizerParameters'], 'CommonOptimizerParameters']

    def __post_init__(self):
        # TODO add checks
        #      for node names
        #      for types
        pass

    def run(self, parameters: 'CommonOptimizerParameters'):
        task = self.task_builder(parameters)
        results = self.runner.run(nodes=self.nodes, task=task, scheme=self.scheme, stop_fun=self.stop_fun)
        return self.parameter_updater(results, parameters)
