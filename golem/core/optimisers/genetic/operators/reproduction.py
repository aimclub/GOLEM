from itertools import cycle
from random import choice
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from golem.core.constants import MIN_POP_SIZE, EVALUATION_ATTEMPTS_NUMBER
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from golem.utilities.data_structures import ensure_wrapped_in_sequence


class ReproductionController:
    """
    Task of the Reproduction Controller is to reproduce population
    while keeping population size as specified in optimizer settings.

    Args:
        parameters: genetic algorithm parameters.
        selection: operator used in reproduction.
        mutation: operator used in reproduction.
        crossover: operator used in reproduction.
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

        self._log = default_log(self)

    def reproduce(self,
                  population: PopulationT,
                  evaluator: EvaluationOperator
                  ) -> PopulationT:
        """Reproduces and evaluates population (select, crossover, mutate).
        Implements additional checks on population to ensure that population size
        follows required population size.
        """
        selected_individuals = self.selection(population, self.parameters.pop_size)
        population_after_crossover = self.crossover(selected_individuals)

        def mutation_n_evaluation(individual: Individual):
            individual = self.mutation(individual)
            if individual:
                individuals = evaluator([individual])
                if individuals:
                    individual = individuals[0]
            return individual

        with Parallel(n_jobs=self.mutation.requirements.n_jobs, prefer='processes', return_as='generator') as parallel:
            new_ind_generator = parallel(delayed(mutation_n_evaluation)(ind)
                                         for ind in population_after_crossover * EVALUATION_ATTEMPTS_NUMBER)

            new_population, pop_graphs = [], []
            for new_ind in new_ind_generator:
                if new_ind and new_ind.graph not in pop_graphs:
                    new_population.append(new_ind)
                    pop_graphs.append(new_ind.graph)
                    if len(new_population) == self.parameters.pop_size:
                        break

        if len(new_population) >= self.parameters.pop_size * self.parameters.required_valid_ratio:
                self._log.info(f'Reproduction achieved pop size {len(new_population)}')
                return new_population
        else:
            # If number of evaluation attempts is exceeded return a warning or raise exception
            helpful_msg = ('Check objective, constraints and evo operators. '
                           'Possibly they return too few valid individuals.')

            if len(new_population) >= self.parameters.pop_size * self._minimum_valid_ratio:
                self._log.warning(f'Could not achieve required population size: '
                                  f'have {len(new_population)},'
                                  f' required {self.parameters.pop_size}!\n' + helpful_msg)
                return new_population
            else:
                raise EvaluationAttemptsError('Could not collect valid individuals'
                                              ' for next population.' + helpful_msg)
