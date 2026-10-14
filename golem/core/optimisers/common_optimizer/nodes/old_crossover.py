from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.genetic.operators.crossover import Crossover as OldCrossover
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin


class CrossoverTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.requirements = parameters.requirements
        self.graph_generation_params = parameters.graph_generation_params
        self.generation = parameters.population

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Crossover(Node):
    def __init__(self, name: str = 'crossover'):
        self.name = name
        self._crossover = None

    def __call__(self, task: CrossoverTask):
        if self._crossover is None:
            self._crossover = OldCrossover(task.graph_optimizer_params,
                                           task.requirements,
                                           task.graph_generation_params)
        task.generation = self._crossover(task.generation)
        task.status = TaskStatusEnum.SUCCESS

        new_tasks = list()
        for new_ind in task.generation:
            new_task = task.copy()
            new_task.generation = [new_ind]
            new_tasks.append(new_task)
        return new_tasks
