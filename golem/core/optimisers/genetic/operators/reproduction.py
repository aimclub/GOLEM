from typing import Optional

import numpy as np

from golem.core.constants import MIN_POP_SIZE, EVALUATION_ATTEMPTS_NUMBER
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover
from golem.core.optimisers.genetic.operators.mutation import Mutation
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError


class ReproductionController:
    """
    Task of the Reproduction Controller is to reproduce population
    while keeping population size as specified in optimizer settings.

    It implements a simple proportional controller that compensates for
    invalid results each generation by computing average ratio of valid results.
    Invalid results include cases when Operators, Evaluator or GraphVerifier
    return output population that's smaller than the input population.

    Args:
        parameters: genetic algorithm parameters.
        selection: operator used in reproduction.
        mutation: operator used in reproduction.
        crossover: operator used in reproduction.
        window_size: size in iterations of the moving window
        to compute reproduction success rate.
    """

    def __init__(self,
                 parameters: GPAlgorithmParameters,
                 selection: Selection,
                 mutation: Mutation,
                 crossover: Crossover,
                 window_size: int = 10,
                 ):
        self.parameters = parameters
        self.selection = selection
        self.mutation = mutation
        self.crossover = crossover

        self._minimum_valid_ratio = parameters.required_valid_ratio * 0.5
        self._window_size = window_size
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
        # TODO: it can't choose more than len(population)!
        #  It can be faster if it could.
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
            residual_size = min(len(population), residual_size)

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
                self._log.info(f'Reproduction achieved pop size {len(next_population)}'
                               f' using {i+1} attempt(s) with success rate {self.mean_success_rate:.3f}')
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
