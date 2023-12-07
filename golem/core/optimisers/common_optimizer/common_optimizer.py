from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Any, Dict, List, Callable

from golem.core.dag.graph import Graph
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.scheme import Scheme
from golem.core.optimisers.common_optimizer.stage import Stage
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum
from golem.core.optimisers.genetic.operators.operator import PopulationT
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import Objective, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimization_parameters import OptimizationParameters
from golem.core.optimisers.optimizer import GraphOptimizer, GraphGenerationParams, AlgorithmParameters
from golem.core.optimisers.timer import OptimisationTimer


@dataclass
class CommonOptimizerParameters:
    _run: bool
    generations: List[PopulationT]

    objective: Objective
    initial_graphs: Sequence[Union[Graph, Any]]
    requirements: OptimizationParameters
    graph_generation_params: GraphGenerationParams
    graph_optimizer_params: AlgorithmParameters
    stages: List[Stage]
    history: OptHistory


class CommonOptimizer(GraphOptimizer):
    __parameters_attrs = ('objective', 'initial_graphs', 'requirements', 'graph_generation_params',
                          'graph_optimizer_params', 'history', 'stages', '_run', 'generations')
    __parameters_allowed_to_change = ('requirements', 'graph_generation_params',
                                      'graph_optimizer_params', 'stages', '_run', 'generations')

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

        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.generations = list()
        self.stages = stages
        self._run = True

    @property
    def parameters(self):
        return CommonOptimizerParameters(**{attr: getattr(self, attr) for attr in self.__parameters_attrs})

    @parameters.setter
    def parameters(self, parameters: CommonOptimizerParameters):
        if not isinstance(parameters, CommonOptimizerParameters):
            raise TypeError(f"parameters should be `CommonOptimizerParameters`, got {type(parameters)} instead")
        for attr in self.__parameters_allowed_to_change:
            setattr(self, attr, getattr(parameters, attr))

    def optimise(self, objective: ObjectiveFunction):
        while self._run:
            for i_stage in range(len(self.stages)):
                self.parameters = self.stages[i_stage].run(self.parameters)
