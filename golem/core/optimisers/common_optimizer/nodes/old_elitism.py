from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.elitism import Elitism as OldElitism



class ElitismTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.generation = parameters.population
        self.best_individuals = None  # TODO

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Elitism(Node):
    def __init__(self, name: str = 'elitism'):
        self.name = name
        self._regularization = None

    def __call__(self, task: ElitismTask):
        if self._regularization is None:
            self._regularization = OldElitism(task.graph_optimizer_params)
        task.generation = self._regularization(task.best_individuals, task.generation)
        task.status = TaskStatusEnum.SUCCESS
        return [task]
