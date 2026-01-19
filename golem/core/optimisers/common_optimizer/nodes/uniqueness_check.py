from copy import deepcopy
from random import choice

from typing import Optional, Dict, Callable

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.base_mutations import base_mutations_repo, MutationTypesEnum
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator


class UniquenessCheckTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.generation = parameters.population
        self.repo = parameters.graph_optimizer_params.repo

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class UniquenessCheck(Node):
    def __init__(self, name: str = 'uniqueness_check'):
        self.name = name

    def __call__(self, task: UniquenessCheckTask):
        to_add = dict()
        new_inds = []
        for ind in task.generation:
            descriptive_id = ind.graph.descriptive_id
            if descriptive_id not in task.repo and descriptive_id not in to_add:
                to_add[descriptive_id] = True
                new_inds.append(ind)

        if to_add:
            task.repo.update(to_add)

        if new_inds:
            task.generation = new_inds
            task.status = TaskStatusEnum.SUCCESS
            return [task]

        task.status = TaskStatusEnum.FAIL
        return [task]

