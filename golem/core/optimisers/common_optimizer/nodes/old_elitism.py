from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.elitism import Elitism as OldElitism



class ElitismTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.new_population = parameters.new_population
        self.best_individuals = parameters.generations.best_individuals

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Elitism(Node):
    def __init__(self, name: str = 'elitism'):
        self.name = name
        self._regularization = None

    def __call__(self, task: ElitismTask):
        if self._regularization is None:
            self._regularization = OldElitism(task.graph_optimizer_params)
        task.new_population = self._regularization(task.best_individuals, task.new_population)
        task.status = TaskStatusEnum.SUCCESS
        return [task]
