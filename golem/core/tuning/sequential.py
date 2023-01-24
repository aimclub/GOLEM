from datetime import timedelta
from functools import partial
from typing import Callable, Optional

from hyperopt import tpe, fmin, space_eval

from golem.core.adapter import BaseOptimizationAdapter
from golem.core.adapter.adapter import DomainStructureType
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.objective import ObjectiveEvaluate
from golem.core.tuning.search_space import SearchSpace, convert_params
from golem.core.tuning.tuner_interface import HyperoptTuner


class SequentialTuner(HyperoptTuner):
    """
    Class for hyperparameters optimization for all nodes sequentially
    """

    def __init__(self, objective_evaluate: ObjectiveEvaluate,
                 search_space: SearchSpace,
                 adapter: Optional[BaseOptimizationAdapter] = None,
                 iterations=100, early_stopping_rounds=None,
                 timeout: timedelta = timedelta(minutes=5),
                 inverse_node_order=False,
                 algo: Callable = tpe.suggest,
                 n_jobs: int = -1):
        super().__init__(objective_evaluate, search_space, adapter, iterations, early_stopping_rounds,
                         timeout, algo, n_jobs)

        self.inverse_node_order = inverse_node_order

    def tune(self, graph: DomainStructureType) -> DomainStructureType:
        """ Method for hyperparameters tuning on the entire graph

        Args:
            graph: graph which hyperparameters will be tuned
        """
        graph = self.adapter.adapt(graph)

        # Check source metrics for data
        self.init_check(graph)

        # Calculate amount of iterations we can apply per node
        nodes_amount = graph.length
        iterations_per_node = round(self.iterations / nodes_amount)
        iterations_per_node = int(iterations_per_node)
        if iterations_per_node == 0:
            iterations_per_node = 1

        # Calculate amount of seconds we can apply per node
        if self.max_seconds is not None:
            seconds_per_node = round(self.max_seconds / nodes_amount)
            seconds_per_node = int(seconds_per_node)
        else:
            seconds_per_node = None

        # Tuning performed sequentially for every node - so get ids of nodes
        nodes_ids = self.get_nodes_order(nodes_number=nodes_amount)
        for node_id in nodes_ids:
            node = graph.nodes[node_id]
            operation_name = node.name

            # Get node's parameters to optimize
            node_params = self.search_space.get_node_params(node_id=node_id,
                                                            operation_name=operation_name)

            if node_params is None:
                self.log.info(f'"{operation_name}" operation has no parameters to optimize')
            else:
                # Apply tuning for current node
                self._optimize_node(node_id=node_id,
                                    graph=graph,
                                    node_params=node_params,
                                    iterations_per_node=iterations_per_node,
                                    seconds_per_node=seconds_per_node)

        # Validation is the optimization do well
        final_graph = self.final_check(graph)

        return self.adapter.restore(final_graph)

    def get_nodes_order(self, nodes_number: int) -> range:
        """ Method returns list with indices of nodes in the graph

        Args:
            nodes_number: number of nodes to get
        """

        if self.inverse_node_order is True:
            # From source data to output
            nodes_ids = range(nodes_number - 1, -1, -1)
        else:
            # From output to source data
            nodes_ids = range(0, nodes_number)

        return nodes_ids

    def tune_node(self, graph: DomainStructureType, node_index: int) -> DomainStructureType:
        """ Method for hyperparameters tuning for particular node

        Args:
            graph: graph which contains a node to be tuned
            node_index: Index of the node to tune

        Returns:
            Graph with tuned parameters in node with specified index
        """
        graph = self.adapter.adapt(graph)

        self.init_check(graph)

        node = graph.nodes[node_index]
        operation_name = node.name

        # Get node's parameters to optimize
        node_params = self.search_space.get_node_params(node_id=node_index,
                                                        operation_name=operation_name)

        if node_params is None:
            self._stop_tuning_with_message(f'"{operation_name}" operation has no parameters to optimize')
        else:
            # Apply tuning for current node
            self._optimize_node(graph=graph,
                                node_id=node_index,
                                node_params=node_params,
                                iterations_per_node=self.iterations,
                                seconds_per_node=self.max_seconds,
                                )

        # Validation is the optimization do well
        final_graph = self.final_check(graph)
        final_graph = self.adapter.restore(final_graph)
        return final_graph

    def _optimize_node(self, graph: OptGraph, node_id: int, node_params: dict, iterations_per_node: int,
                       seconds_per_node: int) -> OptGraph:
        """
        Method for node optimization

        Args:
            graph: Graph which node is optimized
            node_id: id of the current node in the graph
            node_params: dictionary with parameters for node
            iterations_per_node: amount of iterations to produce
            seconds_per_node: amount of seconds to produce

        Returns:
            updated graph with tuned parameters in particular node
        """
        best_parameters = fmin(partial(self._objective,
                                       graph=graph,
                                       node_id=node_id
                                       ),
                               node_params,
                               algo=self.algo,
                               max_evals=iterations_per_node,
                               early_stop_fn=self.early_stop_fn,
                               timeout=seconds_per_node)

        best_parameters = space_eval(space=node_params,
                                     hp_assignment=best_parameters)

        # Set best params for this node in the graph
        graph = self.set_arg_node(graph=graph,
                                  node_id=node_id,
                                  node_params=best_parameters)
        return graph

    def _objective(self, node_params: dict, graph: OptGraph, node_id: int) -> float:
        """ Objective function for minimization problem

        Args:
            node_params: dictionary with parameters for node
            graph: graph to evaluate
            node_id: id of the node to which parameters should be assigned

        Returns:
            value of objective function
        """

        # Set hyperparameters for node
        graph = self.set_arg_node(graph=graph, node_id=node_id,
                                  node_params=node_params)

        metric_value = self.get_metric_value(graph=graph)
        return metric_value

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
        node_params = convert_params(node_params)

        # Update parameters in nodes
        graph.nodes[node_id].parameters = node_params

        return graph
