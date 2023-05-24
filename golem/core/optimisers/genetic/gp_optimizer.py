import math
from copy import deepcopy
from random import choice
from typing import Sequence, Union, Any, Optional

import numpy as np

from golem.core.constants import MAX_GRAPH_GEN_ATTEMPTS, EVALUATION_ATTEMPTS_NUMBER, MIN_POP_SIZE
from golem.core.dag.graph import Graph
from golem.core.log import default_log
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
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer, EvaluationAttemptsError


class EvoGraphOptimizer(PopulationalOptimizer):
    """
    Multi-objective evolutionary graph optimizer named GPComp
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Union[Graph, Any]],
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
        self.reproducer = ReproductionController(graph_optimizer_params, self.selection, self.mutation, self.crossover)

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
        self.initial_individuals = [Individual(graph, metadata=requirements.static_individual_metadata)
                                    for graph in self.initial_graphs]

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        self._update_population(evaluator(self.initial_individuals), 'initial_assumptions')

        if len(self.initial_individuals) < self.graph_optimizer_params.pop_size:
            self.initial_individuals = self._extend_population(self.initial_individuals)
            # Adding of extended population to history
            self._update_population(evaluator(self.initial_individuals), 'extended_initial_assumptions')

    def _extend_population(self, initial_individuals: PopulationT) -> PopulationT:
        initial_individuals = list(initial_individuals)
        initial_graphs = [ind.graph for ind in initial_individuals]
        initial_req = deepcopy(self.requirements)
        initial_req.mutation_prob = 1
        self.mutation.update_requirements(requirements=initial_req)

        for iter_num in range(MAX_GRAPH_GEN_ATTEMPTS):
            if len(initial_individuals) == self.graph_optimizer_params.pop_size:
                break
            new_ind = self.mutation(choice(initial_individuals))
            new_graph = new_ind.graph
            if new_graph not in initial_graphs:
                initial_individuals.append(new_ind)
                initial_graphs.append(new_graph)
        else:
            self.log.warning(f'Exceeded max number of attempts for extending initial graphs, stopping. '
                             f'Current size {len(initial_individuals)} '
                             f'instead of {self.graph_optimizer_params.pop_size} graphs.')

        self.mutation.update_requirements(requirements=self.requirements)
        return initial_individuals

    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """

        # Defines adaptive changes to algorithm parameters
        #  like pop_size and operator probabilities
        self._update_requirements()

        # Regularize previous population
        individuals_to_select = self.regularization(self.population, evaluator)
        # Reproduce from previous pop to get next population
        new_population = self.reproducer.reproduce(individuals_to_select, evaluator)

        # Adaptive agent experience collection & learning
        # Must be called after reproduction (that collects the new experience)
        experience = self.mutation.agent_experience
        experience.collect_results(new_population)
        self.mutation.agent.partial_fit(experience)

        # Use some part of previous pop in the next pop
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


class ReproductionController:
    """
    Task of the Reproduction Controller is to reproduce population
    while keeping population size as specified in optimizer settings.

    It implements a simple proportional controller
    that compensates for invalid results each generation
    by computing average ratio of valid results.
    """

    def __init__(self,
                 parameters: GPAlgorithmParameters,
                 selection: Selection,
                 mutation: Mutation,
                 crossover: Crossover,
                 ):
        self.parameters = parameters
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

        self._minimum_valid_ratio = parameters.required_valid_ratio * 0.5
        self._window_size = max(MIN_POP_SIZE, parameters.max_pop_size // 10)
        self._success_rate_window = np.full(self._window_size, 1.0)

        self._log = default_log(self)

    @property
    def mean_success_rate(self) -> float:
        return float(np.mean(self._success_rate_window))

    def reproduce_uncontrolled(self,
                               population: PopulationT,
                               evaluator: EvaluationOperator,
                               pop_size: Optional[int] = None,
                               ) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
        Doesn't implement any additional checks on population.
        """
        selected_individuals = self.selection(population, pop_size)
        new_population = self.crossover(selected_individuals)
        new_population = self.mutation(new_population)
        new_population = evaluator(new_population)
        return new_population

    def reproduce(self,
                  population: PopulationT,
                  evaluator: EvaluationOperator
                  ) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
        Implements additional checks on population to ensure that population size
        follows required population size.
        """
        required_size = self.parameters.pop_size  # next population size
        next_population = []
        for i in range(EVALUATION_ATTEMPTS_NUMBER):
            # Estimate how many individuals we need to complete new population
            # based on average success rate of valid results
            residual_size = required_size - len(next_population)
            residual_size = max(MIN_POP_SIZE,
                                int(residual_size / self.mean_success_rate))

            # Reproduce the required number of individuals
            new_population = self.reproduce_uncontrolled(population, evaluator, residual_size)
            next_population.extend(new_population)

            # Keep running average of transform success rate (if sample is big enough)
            if len(new_population) > MIN_POP_SIZE:
                valid_ratio = len(new_population) / residual_size
                self._success_rate_window = np.roll(self._success_rate_window, shift=1)
                self._success_rate_window[0] = valid_ratio

            # Successful return: got enough individuals
            if len(next_population) >= required_size * self.parameters.required_valid_ratio:
                return next_population
        else:
            # If number of evaluation attempts is exceeded return a warning or raise exception
            helpful_msg = ('Check objective, constraints and evo operators. '
                           'Possibly they return too few valid individuals.')

            if len(next_population) >= required_size * self._minimum_valid_ratio:
                self._log.warning(f'Could not achieve required population size: '
                                  f'have {len(next_population)}, required {required_size}!\n'
                                  + helpful_msg)
            else:
                raise EvaluationAttemptsError('Could not collect valid individuals'
                                              ' for next population.' + helpful_msg)
