from copy import deepcopy
from random import choice

from typing import Optional, Dict, Callable

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.base_mutations import base_mutations_repo, MutationTypesEnum
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator


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
    def __init__(self,
                 name: str = 'mutation',
                 mutations_repo: Optional[Dict[str], Callable[[OptGraph], OptGraph]] = None):
        self.name = name
        self._mutations_repo = mutations_repo or base_mutations_repo

    def __call__(self, task: MutationTask):
        individual = task.generation[0]
        mutation_type, mutation = choice(self._mutations_repo.items())
        for _ in range(2):
            new_graph = mutation(deepcopy(individual.graph))
            if task.graph_generation_params.verifier(new_graph):
                parent_operator = ParentOperator(type_='mutation',
                                                 operators=mutation_type,
                                                 parent_individuals=individual)
                new_individual = Individual(new_graph, parent_operator,
                                            metadata=task.static_individual_metadata)
                task.generation = [new_individual]
                task.status = TaskStatusEnum.SUCCESS
                return [task]
        task.status = TaskStatusEnum.FAIL
        return [task]
