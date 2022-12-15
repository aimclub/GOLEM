from random import choice
from typing import Union, Optional, Sequence

from golem.core.dag.graph import Graph
from golem.core.optimisers.genetic.evaluation import SimpleDispatcher
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationTypesEnum
from golem.core.optimisers.genetic.operators.operator import EvaluationOperator
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphOptimizer, GraphGenerationParams
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.utilities.grouped_condition import GroupedCondition


class RandomMutationSearchOptimizer(GraphOptimizer):
    """
    Random search-based graph models optimizer
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Union[Graph, Sequence[Graph]],
                 requirements: Optional[GraphRequirements] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_parameters: Optional[GPAlgorithmParameters] = None):
        requirements = requirements or GraphRequirements()
        graph_optimizer_parameters = graph_optimizer_parameters or GPAlgorithmParameters()
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_parameters)
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

    def optimise(self, objective: ObjectiveFunction) -> Graph:

        dispatcher = SimpleDispatcher(self.graph_generation_params.adapter)
        evaluator = dispatcher.dispatch(objective, self.timer)

        self.current_iteration_num = 0

        with self.timer as t:
            best = self._eval_initial_individual(evaluator)
            while not self.stop_optimization():
                mutation = Mutation(self.graph_optimizer_params, self.requirements, self.graph_generation_params)
                new = mutation(best)
                evaluator([new])
                if new.fitness > best.fitness:
                    best = new
                    self.log.info(f'Spent time: {round(self.timer.minutes_from_start, 1)} min')
                    self.log.info(f'Iteration num: {self.current_iteration_num}')
                    self.log.info(f'Best individual fitness: {self._objective.format_fitness(best.fitness)}')

                self.history.add_to_history([best])

                self.current_iteration_num += 1

        return best.graph

    def _eval_initial_individual(self, evaluator: EvaluationOperator) -> Individual:
        initial_individuals = [Individual(graph) for graph in self.initial_graphs]
        best = choice(initial_individuals)
        evaluator([best])
        self.history.add_to_history([best])
        return best
