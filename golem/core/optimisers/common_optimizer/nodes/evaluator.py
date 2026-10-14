from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.mutation import Mutation as OldMutation


class EvaluatorTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.evaluator = parameters.evaluator
        self.generation = parameters.population

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)

from golem.core.optimisers.fitness import null_fitness
class Evaluator(Node):
    def __init__(self, name: str = 'evaluator'):
        self.name = name

    def __call__(self, task: EvaluatorTask):
        evaluated_inds = task.evaluator(task.generation)
        if evaluated_inds:
            task.generation = evaluated_inds
            task.status = TaskStatusEnum.SUCCESS
        else:
            task.status = TaskStatusEnum.FAIL
        return [task]
