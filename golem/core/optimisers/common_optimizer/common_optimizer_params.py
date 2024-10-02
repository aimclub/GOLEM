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
class CommonOptimizerParameters(AlgorithmParameters):
    """ This class is for storing a state of all CommonOptimizer parameters """
    generations: Optional[List[PopulationT]] = None
    population: Optional[PopulationT] = None

    objective: Optional[Objective] = None
    initial_graphs: Optional[Sequence[Union[Graph, Any]]] = None
    requirements: Optional[OptimizationParameters] = None
    graph_generation_params: Optional[GraphGenerationParams] = None
    graph_optimizer_params: Optional[AlgorithmParameters] = None
    history: Optional[OptHistory] = None

    repo: Optional[Dict[str, bool]] = None
    new_population: Optional[PopulationT] = None
    evaluator: Optional[Any] = None

