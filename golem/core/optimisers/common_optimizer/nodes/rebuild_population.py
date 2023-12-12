from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import TaskMixin
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator


class PopulationRebuilderTask(TaskMixin):
    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__(parameters)
        self.static_individual_metadata = parameters.requirements.static_individual_metadata
        self.origin_population = parameters.population
        self.new_population = parameters.new_population

    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        parameters.new_population = self.new_population
        return parameters


class PopulationRebuilder(Node):
    """ Rebuild all individuals in population in main thread """

    def __init__(self, name: str = 'population_rebuilder'):
        self.name = name

    def __call__(self, task: PopulationRebuilderTask):
        uid_to_individual_map = {ind.uid: ind for ind in task.origin_population}

        def rebuild_individual(individual: Individual):
            if individual.uid in uid_to_individual_map:
                # if individual is known, then no need to rebuild it
                new_individual = uid_to_individual_map[individual.uid]
            else:
                parent_operator = None
                if individual.parent_operator:
                    operator = individual.parent_operator
                    parent_individuals = [rebuild_individual(ind) for ind in operator.parent_individuals]
                    parent_operator = ParentOperator(type_=operator.type_,
                                                     operators=operator.operators,
                                                     parent_individuals=parent_individuals)

                new_individual = Individual(individual.graph,
                                            parent_operator,
                                            fitness=individual.fitness,
                                            metadata=task.static_individual_metadata)
                # add new individual to known individuals
                uid_to_individual_map[individual.uid] = new_individual
            return new_individual

        task.new_population = [rebuild_individual(ind) for ind in task.new_population]
        return [task]
