from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.mutation import Mutation as OldMutation


class MutationTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.requirements = parameters.requirements
        self.graph_generation_params = parameters.graph_generation_params
        self.generation = parameters.population

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Mutation(Node):
    def __init__(self, name: str = 'mutation'):
        self.name = name
        self._mutation = None

    def __call__(self, task: MutationTask):
        if self._mutation is None:
            self._mutation = OldMutation(task.graph_optimizer_params,
                                         task.requirements,
                                         task.graph_generation_params)
        ind = self._mutation(task.generation)
        if not ind:
            task.status = TaskStatusEnum.FAIL
        else:
            task.generation = [ind]
            task.status = TaskStatusEnum.SUCCESS
        return [task]
