from copy import deepcopy
from random import choice, random
from typing import Callable, List, Union, Tuple, TYPE_CHECKING, Mapping, Hashable, Optional, Sequence

import numpy as np

from golem.core.dag.graph import Graph
from golem.core.optimisers.adaptive.operatoragent import OperatorAgent, RandomAgent, ExperienceBuffer
from golem.core.optimisers.genetic.operators.base_mutations import base_mutations_repo, MutationTypesEnum, MutationStrengthEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT, Operator
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters
from golem.core.utilities.data_structures import unzip

if TYPE_CHECKING:
    from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters


MutationFunc = Callable[[Graph, GraphRequirements, GraphGenerationParams, AlgorithmParameters], Graph]
MutationIdType = Hashable
MutationRepo = Mapping[MutationIdType, MutationFunc]


class Mutation(Operator):
    def __init__(self,
                 parameters: 'GPAlgorithmParameters',
                 requirements: GraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 mutations_repo: Optional[MutationRepo] = None,
                 operator_agent: Optional[OperatorAgent] = None,
                 ):
        super().__init__(parameters, requirements)
        self.graph_generation_params = graph_gen_params
        self.parameters = parameters
        self._mutations_repo = mutations_repo or base_mutations_repo
        self._operator_agent = operator_agent or RandomAgent(actions=self.parameters.mutation_types)
        self.agent_experience = ExperienceBuffer()

    @property
    def agent(self) -> OperatorAgent:
        return self._operator_agent

    def __call__(self, population: Union[Individual, PopulationT]) -> Union[Individual, PopulationT]:
        if isinstance(population, Individual):
            return self._mutation(population)[0]
        mutated_population, mutations_applied = unzip(map(self._mutation, population))
        self.agent_experience.log_actions(population, mutations_applied)
        return mutated_population

    def _mutation(self, individual: Individual) -> Tuple[Individual, Optional[MutationIdType]]:
        """ Function applies mutation operator to graph """

        for _ in range(self.parameters.max_num_of_operator_attempts):
            new_graph = deepcopy(individual.graph)

            new_graph, mutation_applied = self._apply_mutations(new_graph)

            is_correct_graph = self.graph_generation_params.verifier(new_graph)
            if is_correct_graph:
                parent_operator = ParentOperator(type_='mutation',
                                                 operators=str(mutation_applied),
                                                 parent_individuals=individual)
                return Individual(new_graph, parent_operator,
                                  metadata=self.requirements.static_individual_metadata), mutation_applied

        self.log.debug('Number of mutation attempts exceeded. '
                       'Please check optimization parameters for correctness.')

        return individual, None

    def _sample_num_of_mutations(self) -> int:
        # most of the time returns 1 or rarely several mutations
        if self.parameters.variable_mutation_num:
            num_mut = max(int(round(np.random.lognormal(0, sigma=0.5))), 1)
        else:
            num_mut = 1
        return num_mut

    def _apply_mutations(self, new_graph: OptGraph) -> Tuple[OptGraph, Optional[MutationIdType]]:
        """Apply mutation 1 or few times iteratively"""
        mutation_type = self._operator_agent.choose_action(new_graph)
        mutation_applied = None
        for _ in range(self._sample_num_of_mutations()):
            new_graph, applied = self._adapt_and_apply_mutation(new_graph, mutation_type)
            if applied:
                mutation_applied = mutation_type
                is_custom_mutation = isinstance(mutation_type, Callable)
                if is_custom_mutation:  # custom mutation occurs once
                    break
        return new_graph, mutation_applied

    def _adapt_and_apply_mutation(self, new_graph: OptGraph, mutation_type) -> Tuple[OptGraph, bool]:
        applied = self._will_mutation_be_applied(mutation_type)
        if applied:
            # get the mutation function and adapt it
            mutation_func = self._get_mutation_func(mutation_type)
            new_graph = mutation_func(new_graph, requirements=self.requirements,
                                      graph_gen_params=self.graph_generation_params,
                                      parameters=self.parameters)
            # TODO: add result of the mutation? Optional result?
        return new_graph, applied


    def _will_mutation_be_applied(self, mutation_type: Union[MutationTypesEnum, Callable]) -> bool:
        return random() <= self.parameters.mutation_prob and mutation_type is not MutationTypesEnum.none

    def _get_mutation_func(self, mutation_type: Union[MutationTypesEnum, Callable]) -> Callable:
        if isinstance(mutation_type, Callable):
            mutation_func = mutation_type
        else:
            mutation_func = self._mutations_repo[mutation_type]
        adapted_mutation_func = self.graph_generation_params.adapter.adapt_func(mutation_func)
        return adapted_mutation_func
