from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Any, Dict, List, Callable

from golem.core.dag.graph import Graph
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.old_config import default_stages
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.stage import Stage
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimization_parameters import OptimizationParameters
from golem.core.optimisers.optimizer import GraphOptimizer, GraphGenerationParams, AlgorithmParameters
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer
from golem.core.optimisers.timer import OptimisationTimer


@dataclass
class CommonOptimizerParameters:
    """ This class is for storing a state of all CommonOptimizer parameters """
    _run: bool
    generations: List[PopulationT]
    population: PopulationT
    new_population: PopulationT
    evaluator: Any

    objective: Objective
    initial_graphs: Sequence[Union[Graph, Any]]
    requirements: OptimizationParameters
    graph_generation_params: GraphGenerationParams
    graph_optimizer_params: AlgorithmParameters
    stages: List[Stage]
    history: OptHistory


class CommonOptimizer(PopulationalOptimizer):
    """
    This class implements a common optimizer.

    Args:
        objective: objective for optimization
        initial_graphs: graphs which were initialized outside the optimizer
        requirements: implementation-independent requirements for graph optimizer
        graph_generation_params: parameters for new graph generation
        graph_optimizer_params: parameters for specific implementation of graph optimizer
    """
    __parameters_attrs = ('objective', 'initial_graphs', 'requirements', 'graph_generation_params',
                          'graph_optimizer_params', 'history', 'stages', '_run',
                          'generations', 'population', 'evaluator', 'new_population')
    __parameters_allowed_to_change = ('requirements', 'graph_generation_params',
                                      'graph_optimizer_params', 'stages', '_run', 'new_population')

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Optional[Sequence[Union[Graph, Any]]] = None,
                 requirements: Optional[OptimizationParameters] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_params: Optional[AlgorithmParameters] = None,
                 stages: Optional[List[Stage]] = None):

        super().__init__(objective=objective,
                         initial_graphs=initial_graphs,
                         requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         graph_optimizer_params=graph_optimizer_params)

        self.stages = default_stages
        self._run = True

        self.requirements.max_depth = 100  # TODO fix
        self.graph_optimizer_params.pop_size = graph_optimizer_params.pop_size
        self.initial_individuals = [Individual(graph, metadata=requirements.static_individual_metadata)
                                    for graph in self.initial_graphs]

    @property
    def parameters(self):
        return CommonOptimizerParameters(**{attr: getattr(self, attr) for attr in self.__parameters_attrs})

    @parameters.setter
    def parameters(self, parameters: CommonOptimizerParameters):
        if not isinstance(parameters, CommonOptimizerParameters):
            raise TypeError(f"parameters should be `CommonOptimizerParameters`, got {type(parameters)} instead")
        for attr in self.__parameters_allowed_to_change:
            if hasattr(parameters, attr):
                setattr(self, attr, getattr(parameters, attr))

    # def optimise(self, objective: ObjectiveFunction):
    #     while self._run:
    #         for i_stage in range(len(self.stages)):
    #             self.parameters = self.stages[i_stage].run(self.parameters)

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        self._update_population(evaluator(self.initial_individuals), 'initial_assumptions')

    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """
        # TODO add iterations limit

        self.evaluator = evaluator
        self.new_population = None

        for i_stage in range(len(self.stages)):
            self.parameters = self.stages[i_stage].run(self.parameters)
        return self.new_population
