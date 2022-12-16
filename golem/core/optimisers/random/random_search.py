from typing import Optional, Tuple, Sequence

from golem.core.optimisers.fitness import Fitness
from golem.core.optimisers.genetic.evaluation import SimpleDispatcher
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphOptimizer, GraphGenerationParams
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.utilities.grouped_condition import GroupedCondition


class RandomSearchOptimizer(GraphOptimizer):

    def __init__(self, objective: Objective,
                 requirements: GraphRequirements,
                 graph_generation_params: Optional[GraphGenerationParams] = None):
        super().__init__(objective, requirements=requirements, graph_generation_params=graph_generation_params)
        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.current_iteration_num = 0
        self.stop_optimization = \
            GroupedCondition(results_as_message=True).add_condition(
                lambda: self.timer.is_time_limit_reached(self.current_iteration_num),
                'Optimisation stopped: Time limit is reached'
            ).add_condition(
                lambda: requirements.num_of_generations is not None and
                        self.current_iteration_num >= requirements.num_of_generations,
                'Optimisation stopped: Max number of iterations reached')

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:

        dispatcher = SimpleDispatcher(self.graph_generation_params.adapter)
        evaluator = dispatcher.dispatch(objective, self.timer)
        self.current_iteration_num = 0
        with self.timer:
            best_fitness, best_ind = self._init_assumption(evaluator)
            while not self.stop_optimization():
                new_graph = self.graph_generation_params.random_graph_factory(self.requirements)
                new_ind = Individual(new_graph)
                evaluator([new_ind])
                if new_ind.fitness > best_fitness:
                    best_fitness = new_ind.fitness
                    best_graph = new_graph
                if new_ind.fitness.value:
                    self.history.add_to_history([best_ind])
                    self.log.info(f'Spent time: {round(self.timer.minutes_from_start, 1)} min')
                    self.log.info(f'Iter {self.current_iteration_num}: '
                                  f'best fitness {self._objective.format_fitness(best_fitness)},'
                                  f'try {self._objective.format_fitness(new_ind.fitness)} with num nodes {new_graph.length}')
                self.current_iteration_num += 1
        self.history.add_to_history([best_ind], 'final_choices')
        return [best_graph]

    def _init_assumption(self, evaluator) -> Tuple[Fitness, Individual]:
        new_graph = self.graph_generation_params.random_graph_factory(self.requirements)
        new_ind = Individual(new_graph)
        evaluator([new_ind])
        self.history.add_to_history([new_ind], 'initial_assumptions')
        self.log.info(f'Spent time: {round(self.timer.minutes_from_start, 1)} min')
        self.log.info(f'Initial graph fitness: {self._objective.format_fitness(new_ind.fitness)} with num nodes {new_graph.length}')
        return new_ind.fitness, new_ind
