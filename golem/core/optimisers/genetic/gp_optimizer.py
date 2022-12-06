from copy import copy, deepcopy
from random import choice
from typing import Sequence, Callable

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS
from golem.core.dag.graph import Graph
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.elitism import Elitism
from golem.core.optimisers.genetic.operators.inheritance import Inheritance
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.regularization import Regularization
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.genetic.parameters.graph_depth import AdaptiveGraphDepth
from golem.core.optimisers.genetic.parameters.operators_prob import init_adaptive_operators_prob
from golem.core.optimisers.genetic.parameters.population_size import init_adaptive_pop_size, PopulationSize
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer, EvaluationAttemptsError


EVALUATION_ATTEMPTS_NUMBER = 5


class EvoGraphOptimizer(PopulationalOptimizer):
    """
    Multi-objective evolutionary graph optimizer named GPComp
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[OptGraph],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        # Define genetic operators
        self.regularization = Regularization(graph_optimizer_params, graph_generation_params)
        self.selection = Selection(graph_optimizer_params)
        self.crossover = Crossover(graph_optimizer_params, requirements, graph_generation_params)
        self.mutation = Mutation(graph_optimizer_params, requirements, graph_generation_params)
        self.inheritance = Inheritance(graph_optimizer_params, self.selection)
        self.elitism = Elitism(graph_optimizer_params)
        self.operators = [self.regularization, self.selection, self.crossover,
                          self.mutation, self.inheritance, self.elitism]

        # Define adaptive parameters
        self._pop_size: PopulationSize = init_adaptive_pop_size(graph_optimizer_params, self.generations)
        self._operators_prob = init_adaptive_operators_prob(graph_optimizer_params)
        self._graph_depth = AdaptiveGraphDepth(self.generations,
                                               start_depth=requirements.start_depth,
                                               max_depth=requirements.max_depth,
                                               max_stagnation_gens=graph_optimizer_params.adaptive_depth_max_stagnation,
                                               adaptive=graph_optimizer_params.adaptive_depth)

        # Define initial parameters
        self.requirements.max_depth = self._graph_depth.initial
        self.graph_optimizer_params.pop_size = self._pop_size.initial
        self.initial_individuals = [Individual(graph) for graph in initial_graphs]

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        self._update_population(evaluator(self.initial_individuals), 'initial_assumptions')

        if len(self.initial_individuals) < self.graph_optimizer_params.pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals)
            # Adding of extended population to history
            self._update_population(evaluator(self.initial_individuals), 'extended_initial_assumptions')

    def _extend_population(self, initial_individuals: PopulationT) -> PopulationT:
        iter_num = 0
        initial_individuals = list(initial_individuals)
        initial_graphs = [ind.graph for ind in initial_individuals]
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1
        self.mutation.update_requirements(requirements=initial_req)
        while len(initial_individuals) < self.graph_optimizer_params.pop_size:
            new_ind = self.mutation(choice(self.initial_individuals))
            new_graph = new_ind.graph
            iter_num += 1
            if new_graph not in initial_graphs and self.graph_generation_params.verifier(new_graph):
                initial_individuals.append(new_ind)
                initial_graphs.append(new_graph)
            if iter_num > MAX_GRAPH_GEN_ATTEMPTS:
                self.log.warning(f'Exceeded max number of attempts for extending initial graphs, stopping.'
                                 f'Current size {len(self.initial_individuals)} '
                                 f'instead of {self.graph_optimizer_params.pop_size} graphs.')
                break
        self.mutation.update_requirements(requirements=self.requirements)
        return initial_individuals

    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """
        self._update_requirements()

        individuals_to_select = self.regularization(self.population, evaluator)
        selected_individuals = self.selection(individuals_to_select)
        new_population = self._spawn_evaluated_population(selected_individuals=selected_individuals,
                                                          evaluator=evaluator)
        new_population = self.inheritance(self.population, new_population)
        new_population = self.elitism(self.generations.best_individuals, new_population)

        return new_population

    def _update_requirements(self):
        if not self.generations.is_any_improved:
            self.graph_optimizer_params.mutation_prob, self.graph_optimizer_params.crossover_prob = \
                self._operators_prob.next(self.population)
        self.graph_optimizer_params.pop_size = self._pop_size.next(self.population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(
            f'Next population size: {self.graph_optimizer_params.pop_size}; '
            f'max graph depth: {self.requirements.max_depth}')

        # update requirements in operators
        for operator in self.operators:
            operator.update_requirements(self.graph_optimizer_params, self.requirements)

    def _spawn_evaluated_population(self, selected_individuals: PopulationT, evaluator: Callable) -> PopulationT:
        """ Reproduce and evaluate new population. If at least one of received individuals can not be evaluated then
        mutate and evaluate selected individuals until a new population is obtained
        or the number of attempts is exceeded """

        iter_num = 0
        new_population = None
        while not new_population:
            new_population = self.crossover(selected_individuals)
            new_population = self.mutation(new_population)
            new_population = evaluator(new_population)

            if iter_num > EVALUATION_ATTEMPTS_NUMBER:
                break
            iter_num += 1

        if not new_population:
            raise EvaluationAttemptsError()

        return new_population
