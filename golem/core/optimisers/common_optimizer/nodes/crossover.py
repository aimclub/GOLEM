from random import choice
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskStatusEnum, TaskMixin
from golem.core.optimisers.genetic.operators.crossover import subtree_crossover, one_point_crossover, \
    exchange_edges_crossover, exchange_parents_one_crossover, exchange_parents_both_crossover
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator


class CrossoverTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.generation = parameters.generations[-1]
        self.verifier = parameters.graph_generation_params.verifier
        self.max_depth = parameters.graph_generation_params.max_depth
        self.static_individual_metadata = parameters.requirements.static_individual_metadata

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return super().update_parameters(parameters)


class Crossover(Node):
    def __init__(self, name: str):
        self.name = name
        self.crossovers = [subtree_crossover, one_point_crossover,
                           exchange_edges_crossover, exchange_parents_one_crossover,
                           exchange_parents_both_crossover]

    def __call__(self, task: CrossoverTask):
        new_inds = list()
        inds = task.generation
        for ind1, ind2 in zip(inds[::2], inds[1::2]):
            crossover = choice(self.crossovers)
            new_graphs = crossover(ind1.graph.copy(),
                                   ind2.graph.copy(),
                                   max_depth=task.max_depth)
            for new_graph in new_graphs:
                if task.verifier(new_graph):
                    parent_operator = ParentOperator(type_='crossover',
                                                     operators=str(crossover),
                                                     parent_individuals=[ind1, ind2])
                    new_individual = Individual(new_graph, parent_operator,
                                                metadata=task.static_individual_metadata)
                    new_inds.append(new_individual)

        if new_inds:
            new_tasks = list()
            for new_ind in new_inds:
                new_task = task.copy()
                new_task.generation = [new_ind]
                new_task.status = TaskStatusEnum.SUCCESS
                new_tasks.append(new_task)
            return new_tasks
        else:
            task.status = TaskStatusEnum.FAIL
            return [task]
