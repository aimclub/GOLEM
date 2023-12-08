from random import choice
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.base_mutations import base_mutations_repo
from golem.core.optimisers.genetic.operators.mutation import Mutation as OldMutation
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator


class MutationTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.generation = parameters.generations[-1]
        self.mutation_tries = parameters.graph_optimizer_params.mutation_tries
        self.verifier = parameters.graph_generation_params.verifier
        self.static_individual_metadata = parameters.requirements.static_individual_metadata

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Mutation(Node):
    def __init__(self, name: str):
        self.name = name
        self._mutations_repo = base_mutations_repo

    def __call__(self, task: MutationTask):
        individual = task.generation[0]
        mutation_type, mutation = choice(self._mutations_repo.items())
        for _ in range(task.mutation_tries):
            new_graph = mutation(individual.graph.copy())
            if task.verifier(new_graph):
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
