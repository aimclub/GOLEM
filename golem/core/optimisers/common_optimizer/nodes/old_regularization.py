from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.regularization import Regularization as OldRegularization



class RegularizationTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.graph_generation_params = parameters.graph_generation_params
        self.generation = parameters.population
        self.evaluator = parameters.evaluator

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Regularization(Node):
    def __init__(self, name: str = 'regularization'):
        self.name = name
        self._regularization = None

    def __call__(self, task: RegularizationTask):
        if self._regularization is None:
            self._regularization = OldRegularization(task.graph_optimizer_params,
                                                     task.graph_generation_params)
        task.generation = self._regularization(task.generation, task.evaluator)
        task.status = TaskStatusEnum.SUCCESS
        return [task]
