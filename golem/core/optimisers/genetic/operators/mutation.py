from copy import deepcopy
from functools import partial
from random import choice, random
from typing import Callable, List, Union, Tuple, TYPE_CHECKING

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.optimisers.genetic.operators.base_mutations import (
    no_mutation,
    simple_mutation,
    growth_mutation,
    reduce_mutation,
    single_add_mutation,
    single_edge_mutation,
    single_drop_mutation,
    single_change_mutation
)
from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.core.utilities.data_structures import ComparableEnum as Enum

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


MutationFunc = Callable[[Graph, GraphRequirements, GraphGenerationParams, AlgorithmParameters], Graph]


class MutationTypesEnum(Enum):
    simple = 'simple'
    growth = 'growth'
    local_growth = 'local_growth'
    reduce = 'reduce'
    single_add = 'single_add',
    single_change = 'single_change',
    single_drop = 'single_drop',
    single_edge = 'single_edge'

    none = 'none'


class Mutation(Operator):
    def __init__(self,
                 parameters: 'GPAlgorithmParameters',
                 requirements: GraphRequirements,
                 graph_gen_params: GraphGenerationParams):
        super().__init__(parameters, requirements)
        self.graph_generation_params = graph_gen_params

    def __call__(self, population: Union[Individual, PopulationT]) -> Union[Individual, PopulationT]:
        if isinstance(population, Individual):
            return self._mutation(population)
        return list(map(self._mutation, population))

    def _mutation(self, individual: Individual) -> Individual:
        """ Function applies mutation operator to graph """

        for _ in range(self.parameters.max_num_of_operator_attempts):
            new_graph = deepcopy(individual.graph)
            num_mut = max(int(round(np.random.lognormal(0, sigma=0.5))), 1)

            new_graph, mutation_names = self._adapt_and_apply_mutations(new_graph, num_mut)

            is_correct_graph = self.graph_generation_params.verifier(new_graph)
            if is_correct_graph:
                parent_operator = ParentOperator(type_='mutation', operators=tuple(mutation_names),
                                                 parent_individuals=individual)
                return Individual(new_graph, parent_operator,
                                  metadata=self.requirements.static_individual_metadata)

        self.log.debug('Number of mutation attempts exceeded. '
                       'Please check optimization parameters for correctness.')

        return individual

    def _adapt_and_apply_mutations(self, new_graph: OptGraph, num_mut: int) -> Tuple[OptGraph, List[str]]:
        """Apply mutation in several iterations with specific adaptation of each graph"""

        mutation_types = self.parameters.mutation_types
        is_static_mutation_type = random() < self.parameters.static_mutation_prob
        mutation_type = choice(mutation_types)
        mutation_names = []
        for _ in range(num_mut):
            # determine mutation type
            if not is_static_mutation_type:
                mutation_type = choice(mutation_types)
            is_custom_mutation = isinstance(mutation_type, Callable)

            if self._will_mutation_be_applied(mutation_type):
                # get the mutation function and adapt it
                mutation_func = self._get_mutation_func(mutation_type)
                new_graph = mutation_func(new_graph, requirements=self.requirements,
                                          graph_gen_params=self.graph_generation_params,
                                          parameters=self.parameters)
                # log mutation
                mutation_names.append(str(mutation_type))
                if is_custom_mutation:
                    # custom mutation occurs once
                    break
        return new_graph, mutation_names

    def _will_mutation_be_applied(self, mutation_type: Union[MutationTypesEnum, Callable]) -> bool:
        return random() <= self.parameters.mutation_prob and mutation_type is not MutationTypesEnum.none

    def _get_mutation_func(self, mutation_type: Union[MutationTypesEnum, Callable]) -> Callable:
        if isinstance(mutation_type, Callable):
            mutation_func = mutation_type
        else:
            mutation_func = self.mutation_by_type(mutation_type)
        return self.graph_generation_params.adapter.adapt_func(mutation_func)

    def mutation_by_type(self, mutation_type: MutationTypesEnum) -> Callable:
        mutations = {
            MutationTypesEnum.none: no_mutation,
            MutationTypesEnum.simple: simple_mutation,
            MutationTypesEnum.growth: partial(growth_mutation, local_growth=False),
            MutationTypesEnum.local_growth: partial(growth_mutation, local_growth=True),
            MutationTypesEnum.reduce: reduce_mutation,
            MutationTypesEnum.single_add: single_add_mutation,
            MutationTypesEnum.single_edge: single_edge_mutation,
            MutationTypesEnum.single_drop: single_drop_mutation,
            MutationTypesEnum.single_change: single_change_mutation,
        }
        if mutation_type in mutations:
            return mutations[mutation_type]
        else:
            raise ValueError(f'Required mutation type is not found: {mutation_type}')
