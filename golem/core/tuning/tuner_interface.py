from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import TypeVar, Generic, Optional, Union, Sequence

import numpy as np

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.adapter.adapter import IdentityAdapter
from golem.core.constants import MAX_TUNING_METRIC_VALUE, MIN_TIME_FOR_TUNING_IN_SEC
from golem.core.dag.graph_utils import graph_structure
from golem.core.log import default_log
from golem.core.optimisers.fitness import SingleObjFitness, MultiObjFitness
from golem.core.optimisers.fitness.fitness import Fitness
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory, OptHistoryLabels
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.optimisers.timer import Timer
from golem.core.tuning.search_space import SearchSpace, convert_parameters
from golem.utilities.data_structures import ensure_wrapped_in_sequence

DomainGraphForTune = TypeVar('DomainGraphForTune')


class BaseTuner(Generic[DomainGraphForTune]):
    """
    Base class for hyperparameters optimization

    Args:
      objective_evaluate: objective to optimize
      adapter: the function for processing of external object that should be optimized
      search_space: SearchSpace instance
      iterations: max number of iterations
      early_stopping_rounds: Optional max number of stagnating iterations for early stopping.
      timeout: max time for tuning
      n_jobs: num of ``n_jobs`` for parallelization (``-1`` for use all cpu's)
      deviation: required improvement (in percent) of a metric to return tuned graph.
        By default, ``deviation=0.05``, which means that tuned graph will be returned
        if it's metric will be at least 0.05% better than the initial.
      history: optional ``OptHistory``. When given, every graph evaluated during tuning
        (via `evaluate_graph`) is recorded into it as its own generation, each individual
        carrying a `ParentOperator(type_='tuning', ...)` that points back at the individual
        tuning started from (`OptHistoryLabels.tuning_start`). The graph(s) tuning finally
        returns are additionally recorded under `OptHistoryLabels.tuning_results`.
    """

    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05,
                 history: OptHistory = None,
                 **kwargs):
        self.iterations = iterations
        self.adapter = adapter or IdentityAdapter()
        self.search_space = search_space
        self.n_jobs = n_jobs
        if isinstance(objective_evaluate, ObjectiveEvaluate):
            objective_evaluate.eval_n_jobs = self.n_jobs
        self.objective_evaluate = self.adapter.adapt_func(objective_evaluate)
        self.deviation = deviation

        self.timeout = timeout
        self.timer = Timer()
        self.early_stopping_rounds = early_stopping_rounds

        self._default_metric_value = MAX_TUNING_METRIC_VALUE
        self.was_tuned = False
        self.init_graph = None
        self.init_metric = None
        self.obtained_metric = None
        self.log = default_log(self)
        self.objectives_number = 1

        self.history = history
        self.evaluations_count = 0
        self.init_individual: Optional[Individual] = None
        self.obtained_individual: Optional[Individual] = None

    def tune(self, graph: DomainGraphForTune, **kwargs) -> Union[DomainGraphForTune, Sequence[DomainGraphForTune]]:
        """
        Function for hyperparameters tuning on the graph

        Args:
          graph: domain graph for which hyperparameters tuning is needed

        Returns:
          Graph with optimized hyperparameters
          or pareto front of optimized graphs in case of multi-objective optimization
        """
        graph = self.adapter.adapt(graph)
        self.was_tuned = False
        self.evaluations_count = 0
        self.obtained_individual = None
        with self.timer:

            # Check source metrics for data
            self.init_check(graph)
            final_graph = self._tune(graph, **kwargs)
            # Validate if optimisation did well
            final_graph = self.final_check(final_graph, self.objectives_number > 1)

        final_graph = self.adapter.restore(final_graph)
        return final_graph

    @abstractmethod
    def _tune(self, graph: DomainGraphForTune, **kwargs):
        raise NotImplementedError

    def init_check(self, graph: OptGraph) -> None:
        """
        Method get metric on validation set before start optimization

        Args:
          graph: graph to calculate objective
          multi_obj: If optimization was multi objective.
        """
        self.log.info('Hyperparameters optimization start: estimation of metric for initial graph')

        # Train graph
        self.init_graph = deepcopy(graph)
        init_fitness = self.objective_evaluate(self.init_graph)

        # Root individual: no parent, since this is what tuning starts from.
        self.init_individual = self._create_individual(self.init_graph, init_fitness, parent=None)
        self._add_to_history([self.init_individual], OptHistoryLabels.tuning_start)

        self.init_metric = self._fitness_to_metric_value(init_fitness)

        self.log.message(f'Initial graph: {graph_structure(self.init_graph)} \n'
                         f'Initial metric: '
                         f'{list(map(lambda x: round(abs(x), 3), ensure_wrapped_in_sequence(self.init_metric)))}')

    def final_check(self, tuned_graphs: Union[OptGraph, Sequence[OptGraph]], multi_obj: bool = False) \
            -> Union[OptGraph, Sequence[OptGraph]]:
        """
        Method propose final quality check after optimization process

        Args:
          tuned_graphs: Tuned graph to calculate objective
          multi_obj: If optimization was multi objective.
        """
        self.log.info('Hyperparameters optimization finished')

        if multi_obj:
            final_graphs = self._multi_obj_final_check(tuned_graphs)
            self._record_tuning_results(final_graphs, self.obtained_metric)
            return final_graphs
        else:
            final_graph = self._single_obj_final_check(tuned_graphs)
            self._record_tuning_results([final_graph], [self.obtained_metric])
            return final_graph

    def _single_obj_final_check(self, tuned_graph: OptGraph):
        self.obtained_metric = self.get_metric_value(graph=tuned_graph)

        prefix_tuned_phrase = 'Return tuned graph due to the fact that obtained metric'
        prefix_init_phrase = 'Return init graph due to the fact that obtained metric'

        if np.isclose(self.obtained_metric, self._default_metric_value):
            self.obtained_metric = None

        # 0.05% deviation is acceptable
        deviation_value = (self.init_metric / 100.0) * self.deviation
        init_metric = self.init_metric + deviation_value * (-np.sign(self.init_metric))
        if self.obtained_metric is None:
            self.log.info(f'{prefix_init_phrase} is None. Initial metric is {abs(init_metric):.3f}')
            final_graph = self.init_graph
            final_metric = self.init_metric
        elif self.obtained_metric <= init_metric:
            self.log.info(f'{prefix_tuned_phrase} {abs(self.obtained_metric):.3f} equal or '
                          f'better than initial (+ {self.deviation}% deviation) {abs(init_metric):.3f}')
            final_graph = tuned_graph
            final_metric = self.obtained_metric
        else:
            self.log.info(f'{prefix_init_phrase} {abs(self.obtained_metric):.3f} '
                          f'worse than initial (+ {self.deviation}% deviation) {abs(init_metric):.3f}')
            final_graph = self.init_graph
            final_metric = self.init_metric
            self.obtained_metric = final_metric
        self.log.message(f'Final graph: {graph_structure(final_graph)}')
        if final_metric is not None:
            self.log.message(f'Final metric: {abs(final_metric):.3f}')
        else:
            self.log.message('Final metric is None')
        return final_graph

    def _multi_obj_final_check(self, tuned_graphs: Sequence[OptGraph]) -> Sequence[OptGraph]:
        self.obtained_metric = []
        final_graphs = []
        for tuned_graph in tuned_graphs:
            obtained_metric = self.get_metric_value(graph=tuned_graph)
            for e, value in enumerate(obtained_metric):
                if np.isclose(value, self._default_metric_value):
                    obtained_metric[e] = None
            if not MultiObjFitness(self.init_metric).dominates(MultiObjFitness(obtained_metric)):
                self.obtained_metric.append(obtained_metric)
                final_graphs.append(tuned_graph)
        if final_graphs:
            metrics_formatted = [str([round(x, 3) for x in metrics]) for metrics in self.obtained_metric]
            metrics_formatted = '\n'.join(metrics_formatted)
            self.log.message('Return tuned graphs with obtained metrics \n'
                             f'{metrics_formatted}')
        else:
            self.log.message('Initial metric dominates all found solutions. Return initial graph.')
            final_graphs = [self.init_graph]
            self.obtained_metric = [self.init_metric]
        return final_graphs

    def _fitness_to_metric_value(self, graph_fitness: Fitness) -> Union[float, Sequence[float]]:
        if isinstance(graph_fitness, SingleObjFitness):
            if not graph_fitness.valid:
                return self._default_metric_value

            return graph_fitness.value

        if isinstance(graph_fitness, MultiObjFitness):
            return tuple(
                self._default_metric_value if value is None else value
                for value in graph_fitness.values
            )

        raise ValueError(
            f'Objective evaluation must be a Fitness instance, '
            f'not {graph_fitness}.'
        )

    def _metric_to_fitness(self, metric: Union[float, Sequence[float], None]) -> Fitness:
        """ Inverse of `_fitness_to_metric_value`: wraps a metric value (as produced by
        `get_metric_value`/stored in `self.obtained_metric`) back into a `Fitness`, so it
        can be attached to an `Individual` recorded under `OptHistoryLabels.tuning_results`.
        `final_check` only has the scalar metric at that point, not the original `Fitness`
        object, so this (rather than reusing `_fitness_to_metric_value`) is what's needed there.
        """
        if metric is None:
            return SingleObjFitness() if self.objectives_number == 1 else MultiObjFitness()
        if self.objectives_number > 1 or isinstance(metric, (list, tuple)):
            return MultiObjFitness(metric)
        return SingleObjFitness(metric)

    def _create_individual(self, graph: OptGraph, fitness: Fitness,
                           parent: Optional[Individual] = None) -> Individual:
        """
        Args:
          graph: graph the individual wraps.
          fitness: its evaluated fitness.
          parent: the individual this graph was derived from by tuning (i.e. same structure,
            different hyperparameters). Pass ``None`` only for the very first, root individual
            (see `init_check`) — every individual produced *during* tuning must reference the
            individual it started from, or the recorded `ParentOperator` carries no lineage
            and genealogy-based visualizations (e.g. FEDOT.Web) will show it as a disconnected
            root instead of linking it to what it was tuned from.
        """
        parent_individuals = [parent] if parent is not None else []

        parent_operator = ParentOperator(
            type_='tuning',
            operators=self.__class__.__name__,
            parent_individuals=parent_individuals
        )

        return Individual(
            graph=deepcopy(graph),
            parent_operator=parent_operator,
            fitness=fitness
        )

    def _add_to_history(
            self,
            individuals: Sequence[Individual],
            label: Optional[str] = None
    ):
        if self.history is None:
            return

        label = label or f'tuning_iteration_{self.evaluations_count}'

        self.history.add_to_history(
            individuals=individuals,
            generation_label=label,
            generation_metadata={
                'tuner': self.__class__.__name__
            }
        )

    def _record_tuning_results(self, graphs: Sequence[OptGraph], metrics: Sequence) -> None:
        """ Records the graph(s) `tune()` is about to return under `OptHistoryLabels.tuning_results`,
        linked back to `self.init_individual`, so the actual outcome of tuning is unambiguously
        marked — as opposed to the (possibly many) intermediate candidates recorded by
        `evaluate_graph` under generic `tuning_iteration_N` labels.
        """
        if self.history is None or self.init_individual is None:
            return

        result_individuals = [
            self._create_individual(graph, self._metric_to_fitness(metric), parent=self.init_individual)
            for graph, metric in zip(graphs, metrics)
        ]
        self.obtained_individual = result_individuals[0] if len(result_individuals) == 1 else result_individuals
        self._add_to_history(result_individuals, OptHistoryLabels.tuning_results)

    def get_metric_value(self, graph: OptGraph) -> Union[float, Sequence[float]]:
        """
        Method calculates metric for algorithm validation

        Args:
          graph: Graph to evaluate

        Returns:
          value of loss function
        """
        graph_fitness = self.objective_evaluate(graph)

        if isinstance(graph_fitness, SingleObjFitness):
            metric_value = graph_fitness.value
            if not graph_fitness.valid:
                return self._default_metric_value
            return metric_value

        elif isinstance(graph_fitness, MultiObjFitness):
            metric_values = graph_fitness.values
            for e, value in enumerate(metric_values):
                if value is None:
                    metric_values[e] = self._default_metric_value
            return metric_values

    def evaluate_graph(self, graph: OptGraph) -> Union[float, Sequence[float]]:
        """Evaluate a graph and save its state to tuning history."""

        fitness = self.objective_evaluate(graph)

        individual = self._create_individual(graph, fitness, parent=self.init_individual)

        self._add_to_history([individual])

        self.evaluations_count += 1

        return self._fitness_to_metric_value(fitness)

    @staticmethod
    def set_arg_graph(graph: OptGraph, parameters: dict) -> OptGraph:
        """ Method for parameters setting to a graph

        Args:
            graph: graph to which parameters should be assigned
            parameters: dictionary with parameters to set

        Returns:
            graph: graph with new hyperparameters in each node
        """
        # Set hyperparameters for every node
        for node_id, node in enumerate(graph.nodes):
            node_params = {key: value for key, value in parameters.items()
                           if key.startswith(f'{str(node_id)} || {node.name}')}

            if node_params is not None:
                BaseTuner.set_arg_node(graph, node_id, node_params)

        return graph

    @staticmethod
    def set_arg_node(graph: OptGraph, node_id: int, node_params: dict) -> OptGraph:
        """ Method for parameters setting to a node

        Args:
            graph: graph which contains the node
            node_id: id of the node to which parameters should be assigned
            node_params: dictionary with labeled parameters to set

        Returns:
            graph with new hyperparameters in the specified node
        """

        # Remove label prefixes
        node_params = convert_parameters(node_params)

        # Update parameters in the specified node
        graph.nodes[node_id].parameters = node_params

        return graph

    def _check_if_tuning_possible(self, graph: OptGraph,
                                  parameters_to_optimize: bool,
                                  remaining_time: Optional[float] = None,
                                  supports_multi_objective: bool = False) -> bool:
        if len(ensure_wrapped_in_sequence(self.init_metric)) > 1 and not supports_multi_objective:
            self._stop_tuning_with_message(f'{self.__class__.__name__} does not support multi-objective optimization.')
            return False
        elif not parameters_to_optimize:
            self._stop_tuning_with_message(f'Graph "{graph.graph_description}" has no parameters to optimize')
            return False
        elif remaining_time is not None:
            if remaining_time <= MIN_TIME_FOR_TUNING_IN_SEC:
                self._stop_tuning_with_message('Tunner stopped after initial assumption due to the lack of time')
                return False
        return True

    def _stop_tuning_with_message(self, message: str):
        self.log.message(message)
        self.obtained_metric = self.init_metric

    def _get_remaining_time(self) -> Optional[float]:
        if self.timeout is not None:
            remaining_time = self.timeout.seconds - self.timer.seconds_from_start
            return remaining_time
        else:
            return None