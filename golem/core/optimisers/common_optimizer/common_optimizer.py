from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Any, Dict, List

from golem.core.dag.graph import Graph
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.scheme import Scheme
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
    objective: Objective
    initial_graphs: Sequence[Union[Graph, Any]]
    requirements: OptimizationParameters
    graph_generation_params: GraphGenerationParams
    graph_optimizer_params: AlgorithmParameters
    history: OptHistory


class Runner:
    def __init__(self):
        self.history = list()

    @abstractmethod
    def run(self, scheme: Scheme, task: Task):
        raise NotImplementedError('It is abstract method')


class ParallelRunner(Runner):
    def __init__(self):
        super().__init__()


class OneThreadRunner(Runner):
    def __init__(self):
        super().__init__()

    def run(self, scheme: Scheme, task: Task, nodes: Dict[Node]):
        self.history.append(task)
        tasks = deque()
        while task.status is not TaskStatusEnum.FINISH_RUNNER:
            task = scheme.next(task)
            processed_tasks = nodes[task.get_next_node()](task)
            tasks.extend(processed_tasks)
            task = tasks.popleft()
            self.history.append(task)


class CommonOptimizer(GraphOptimizer):
    __parameters_attrs = ('objective', 'initial_graphs', 'requirements', 'graph_generation_params',
                          'graph_optimizer_params', 'history')
    __parameters_allowed_to_change = ('requirements', 'graph_generation_params', 'graph_optimizer_params')

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Optional[Sequence[Union[Graph, Any]]] = None,
                 requirements: Optional[OptimizationParameters] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_params: Optional[AlgorithmParameters] = None):
        super().__init__(objective=objective,
                         initial_graphs=initial_graphs,
                         requirements=requirements,
                         graph_generation_params=graph_generation_params,
                         graph_optimizer_params=graph_optimizer_params)

        self.timer = OptimisationTimer(timeout=self.requirements.timeout)
        self.generations = list()

    @property
    def parameters(self):
        return CommonOptimizerParameters(**{attr: getattr(self, attr) for attr in self.__parameters_attrs})

    @parameters.setter
    def parameters(self, parameters: CommonOptimizerParameters):
        if not isinstance(parameters, CommonOptimizerParameters):
            raise TypeError(f"parameters should be `CommonOptimizerParameters`, got {type(parameters)} instead")
        for attr in self.__parameters_allowed_to_change:
            setattr(self, attr, getattr(parameters, attr))

    def optimise(self, objective: ObjectiveFunction) -> Sequence[OptGraph]:
        parameters = self.parameters
        with self.timer, self._progressbar as pbar:
            while True:
                parameters = self.parameters
                for pool in self.pools:
                    parameters = pool(parameters)
                self._update_population(parameters.population)
        pbar.close()

        self.parameters = parameters

        self._update_population(self.best_individuals, 'final_choices')
        return [ind.graph for ind in self.best_individuals]

    def _update_population(self, next_population: PopulationT, label: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        self.generations.append(next_population)
        if self.requirements.keep_history:
            self._log_to_history(next_population, label, metadata)
        self._iteration_callback(next_population, self)

    def _log_to_history(self, population: PopulationT, label: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        self.history.add_to_history(population, label, metadata)
        self.history.add_to_archive_history(self.generations.best_individuals)
        if self.requirements.history_dir:
            self.history.save_current_results(self.requirements.history_dir)
