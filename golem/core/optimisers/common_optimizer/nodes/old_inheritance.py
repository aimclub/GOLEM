from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.inheritance import Inheritance as OldInheritance
from golem.core.optimisers.genetic.operators.selection import Selection as OldSelection

class InheritanceTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.generation = parameters.population
        self.origin_generation = parameters.population

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Inheritance(Node):
    def __init__(self, name: str = 'inheritance'):
        self.name = name
        self._inheritance = None

    def __call__(self, task: InheritanceTask):
        if self._inheritance is None:
            selection = OldSelection(task.graph_optimizer_params)
            self._inheritance = OldInheritance(task.graph_optimizer_params, selection)
        task.generation = self._inheritance(task.origin_generation, task.generation)
        task.status = TaskStatusEnum.SUCCESS
        return [task]
