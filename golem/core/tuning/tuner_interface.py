from abc import abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import Generic, Optional, Sequence, TypeVar, Union

import numpy as np

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.adapter.adapter import IdentityAdapter
from golem.core.constants import MAX_TUNING_METRIC_VALUE
from golem.core.dag.graph_utils import graph_structure
from golem.core.log import default_log
from golem.core.optimisers.fitness import Fitness, MultiObjFitness, SingleObjFitness
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate, ObjectiveFunction
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory, TUNING_RESULT_LABEL, TUNING_START_LABEL
from golem.core.optimisers.opt_history_objects.parent_operator import ParentOperator
from golem.core.tuning.search_space import SearchSpace, convert_parameters
from golem.core.utilities.data_structures import ensure_wrapped_in_sequence

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
      history: object to store tuning history if needed.
    """

    def __init__(self, objective_evaluate: ObjectiveFunction,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations: int = 100,
                 early_stopping_rounds: Optional[int] = None,
                 timeout: timedelta = timedelta(minutes=5),
                 n_jobs: int = -1,
                 deviation: float = 0.05,
                 history: Optional[OptHistory] = None):
        self.iterations = iterations
        self.evaluations_count = 0
        self.adapter = adapter or IdentityAdapter()
        self.search_space = search_space
        self.n_jobs = n_jobs
        if isinstance(objective_evaluate, ObjectiveEvaluate):
            objective_evaluate.eval_n_jobs = self.n_jobs
        self.objective_evaluate = self.adapter.adapt_func(objective_evaluate)
        self.deviation = deviation

        self.timeout = timeout
        self.early_stopping_rounds = early_stopping_rounds

        self._default_metric_value = MAX_TUNING_METRIC_VALUE
        self.was_tuned = False
        self.init_individual = None
        self.obtained_individual = None
        self.history = history
        self.log = default_log(self)

    @abstractmethod
    def tune(self, graph: DomainGraphForTune) -> Union[DomainGraphForTune, Sequence[DomainGraphForTune]]:
        """
        Function for hyperparameters tuning on the graph

        Args:
          graph: domain graph for which hyperparameters tuning is needed

        Returns:
          Graph with optimized hyperparameters
          or pareto front of optimized graphs in case of multi-objective optimization
        """
        raise NotImplementedError()

    def init_check(self, graph: OptGraph) -> None:
        """
        Method gets metric on validation set before starting optimization

        Args:
          graph: graph to calculate objective
        """
        self.log.info('Hyperparameters optimization start: estimation of metric for initial graph')

        # Train graph
        graph = deepcopy(graph)
        fitness = self.objective_evaluate(graph)
        self.init_individual = self._create_individual(graph, fitness)
        self._add_to_history([self.init_individual], label=TUNING_START_LABEL)

        init_metric = self._fitness_to_metric_value(fitness)
        self.log.message(f'Initial graph: {graph_structure(graph)} \n'
                         f'Initial metric: '
                         f'{list(map(lambda x: round(abs(x), 3), ensure_wrapped_in_sequence(init_metric)))}')

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
            return self._multi_obj_final_check(tuned_graphs)
        else:
            return self._single_obj_final_check(tuned_graphs)

    def _single_obj_final_check(self, tuned_graph: OptGraph):
        obtained_fitness = self.objective_evaluate(tuned_graph)
        init_individual = self.init_individual

        prefix_tuned_phrase = 'Return tuned graph due to the fact that obtained metric'
        prefix_init_phrase = 'Return init graph due to the fact that obtained metric'

        # 0.05% deviation is acceptable
        init_metric = self._fitness_to_metric_value(init_individual.fitness)
        deviation_value = (init_metric / 100.0) * self.deviation
        init_fitness_with_deviation = SingleObjFitness(init_metric + deviation_value * (-np.sign(init_metric)))

        if not obtained_fitness.valid:
            self.log.info(f'{prefix_init_phrase} is None. Initial metric is {abs(init_metric):.3f}')
            final_individual = init_individual
        elif obtained_fitness >= init_fitness_with_deviation:
            obtained_metric = self._fitness_to_metric_value(obtained_fitness)
            self.log.info(f'{prefix_tuned_phrase} {abs(obtained_metric):.3f} equal or '
                          f'better than initial (+ {self.deviation}% deviation) {abs(init_metric):.3f}')
            final_individual = self._create_individual(tuned_graph, obtained_fitness)
        else:
            obtained_metric = self._fitness_to_metric_value(obtained_fitness)
            self.log.info(f'{prefix_init_phrase} {abs(obtained_metric):.3f} '
                          f'worse than initial (+ {self.deviation}% deviation) {abs(init_metric):.3f}')
            final_individual = init_individual
        self.log.message(f'Final graph: {graph_structure(final_individual.graph)}')

        final_metric = self._fitness_to_metric_value(final_individual.fitness)

        if final_metric is not None:
            self.log.message(f'Final metric: {abs(final_metric):.3f}')
        else:
            self.log.message('Final metric is None')

        self.obtained_individual = final_individual
        self._add_to_history([self.obtained_individual], label=TUNING_RESULT_LABEL)

        return self.obtained_individual.graph

    def _multi_obj_final_check(self, tuned_graphs: Sequence[OptGraph]) -> Sequence[OptGraph]:
        obtained_fitnesses = [self.objective_evaluate(graph) for graph in tuned_graphs]

        final_graphs = []
        self.obtained_individual = []
        for tuned_graph, obtained_fitness in zip(tuned_graphs, obtained_fitnesses):
            if obtained_fitness.dominates(self.init_individual.fitness):
                individual = self._create_individual(tuned_graph, obtained_fitness)
                self.obtained_individual.append(individual)
                final_graphs.append(tuned_graph)
        if final_graphs:
            obtained_metrics = [self._fitness_to_metric_value(fitness) for fitness in obtained_fitnesses]
            metrics_formatted = [str([round(x, 3) for x in metrics]) for metrics in obtained_metrics]
            metrics_formatted = '\n'.join(metrics_formatted)
            self.log.message('Return tuned graphs with obtained metrics \n'
                             f'{metrics_formatted}')
        else:
            self.log.message('Initial metric dominates all found solutions. Return initial graph.')
            self.obtained_individual = [self.init_individual]
            final_graphs = [self.init_individual.graph]

        self._add_to_history(self.obtained_individual, label=TUNING_RESULT_LABEL)

        return final_graphs

    def evaluate_graph(self, graph: OptGraph) -> Union[float, Sequence[float]]:
        """
        Method calculates metric for algorithm validation.
        Also, responsible for saving of tuning history.

        Args:
          graph: Graphs to evaluate

        Returns:
          values of loss function for graph
        """
        graph_fitness = self.objective_evaluate(graph)
        individual = self._create_individual(graph, graph_fitness)
        self._add_to_history([individual])
        self.evaluations_count += 1
        return self._fitness_to_metric_value(graph_fitness)

    def _fitness_to_metric_value(self, fitness: Fitness) -> Union[float, Sequence[float]]:
        if isinstance(fitness, SingleObjFitness):
            metric_value = fitness.value
            if not fitness.valid:
                metric_value = self._default_metric_value

        elif isinstance(fitness, MultiObjFitness):
            metric_value = fitness.values
            metric_value = tuple(self._default_metric_value if value is None else value for value in metric_value)
        else:
            raise ValueError(f'Objective evaluation must be a Fitness instance, not {fitness}.')
        return metric_value

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
        """ Method for parameters setting to a graph

        Args:
            graph: graph which contains the node
            node_id: id of the node to which parameters should be assigned
            node_params: dictionary with labeled parameters to set

        Returns:
            graph with new hyperparameters in each node
        """

        # Remove label prefixes
        node_params = convert_parameters(node_params)

        # Update parameters in nodes
        graph.nodes[node_id].parameters = node_params

        return graph

    def _stop_tuning_with_message(self, message: str):
        self.log.message(message)
        self.obtained_fitness = self.init_individual.fitness

    def _create_individual(self, graph: OptGraph, fitness: Fitness) -> Individual:
        history = self.history

        if history and history.tuning_start:
            parent_individuals = history.tuning_start
        elif history and history.evolution_results:
            parent_individuals = history.evolution_results
        else:
            parent_individuals = []
        tuner_name = self.__class__.__name__
        parent_operator = ParentOperator(type_='tuning', operators=[tuner_name],
                                         parent_individuals=parent_individuals)
        individual = Individual(graph, parent_operator=parent_operator, fitness=fitness)

        return individual

    def _add_to_history(self, individuals: Sequence[Individual], label: Optional[str] = None):
        history = self.history
        tuner_name = self.__class__.__name__

        if not history:
            return

        if label is None:
            label = f'tuning_iteration_{self.evaluations_count}'
        if label not in (TUNING_START_LABEL, TUNING_RESULT_LABEL):
            individuals = list(individuals)
            individuals.append(self.init_individual)  # add initial individual to maintain consistency of inheritance
        history.add_to_history(individuals=individuals,
                               generation_label=label,
                               generation_metadata=dict(tuner=tuner_name))
