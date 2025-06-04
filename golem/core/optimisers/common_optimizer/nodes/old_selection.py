from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.selection import Selection as OldSelection



class SelectionTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.generation = parameters.population

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Selection(Node):
    def __init__(self, name: str = 'selection'):
        self.name = name
        self._selection = None

    def __call__(self, task: SelectionTask):
        if self._selection is None:
            self._selection = OldSelection(task.graph_optimizer_params)
        task.generation = self._selection(task.generation)
        task.status = TaskStatusEnum.SUCCESS
        return [task]
