import random
from abc import ABC, abstractmethod
from copy import deepcopy
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type, Union, Dict, Callable, Any

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_node_factory import OptNodeFactory
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.graph_sa.sa_requirements import StructuralAnalysisRequirements, \
    ReplacementAnalysisMetaParams


class NodeAnalysis:
    """
    :param approaches: methods applied to nodes to modify the graph or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceOperationAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, node_factory: Any,
                 approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None,
                 approaches_requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):

        self.node_factory = node_factory

        self.approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze] if approaches is None else approaches

        self.path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        self.log = default_log(self)

        self.approaches_requirements = \
            StructuralAnalysisRequirements() if approaches_requirements is None else approaches_requirements

    def analyze(self, graph: OptGraph, node: OptNode,
                objectives: List[Callable],
                timer: OptimisationTimer = None) -> dict:

        """
        Method runs Node analysis within defined approaches

        :param graph: Graph containing the analyzed Node
        :param node: Node object to analyze in Graph
        :param objectives: objective functions for computing metric values
        :param timer: timer to check if the time allotted for structural analysis has expired
        :return: dict with Node analysis result per approach
        """

        results = dict()
        for approach in self.approaches:
            if timer is not None and timer.is_time_limit_reached():
                results[f'{approach.__name__}'] = {'loss': [-2.0]*len(objectives)}
                break
            results[f'{approach.__name__}'] = \
                approach(graph=graph,
                         objectives=objectives,
                         node_factory=self.node_factory,
                         requirements=self.approaches_requirements,
                         path_to_save=self.path_to_save).analyze(node=node)
        return results


class NodeAnalyzeApproach(ABC):
    """
    Base class for analysis approach.
    :param graph: Graph containing the analyzed Node
    :param objectives: objective functions for computing metric values
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, graph: OptGraph, objectives: List[Callable],
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):
        self._graph = graph
        self._objectives = objectives
        self._node_factory = node_factory
        self._origin_metrics = list()
        self._requirements = \
            StructuralAnalysisRequirements() if requirements is None else requirements

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        self.log = default_log(prefix='node_analysis')

        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    @abstractmethod
    def analyze(self, node: OptNode, **kwargs) -> Union[List[dict], List[float]]:
        """Creates the difference metric(scorer, index, etc) of the changed
        graph in relation to the original one
        :param node: the sequence number of the node as in DFS result
        """
        pass

    @abstractmethod
    def sample(self, *args) -> Union[List[OptGraph], OptGraph]:
        """Changes the graph according to the approach"""
        pass

    def _is_the_modified_graph_different(self, modified_graph: OptGraph):
        """ Checks if the graph after changes is different from the original graph """
        if modified_graph.root_node.descriptive_id != self._graph.root_node.descriptive_id:
            return True
        return False

    def _compare_with_origin_by_metrics(self, modified_graph: OptGraph) -> List[float]:
        """ Iterate through all objectives and evaluate modified graph """
        results = []
        for objective in self._objectives:
            metric = self._compare_with_origin_by_metric(modified_graph=modified_graph,
                                                         objective=objective)
            results.append(metric)
        return results

    def _compare_with_origin_by_metric(self, modified_graph: OptGraph,
                                       objective: Callable) -> float:
        """ Returns the ratio of metrics for the modified graph and the original one """

        if not self._is_the_modified_graph_different(modified_graph):
            return -1.0

        obj_idx = self._objectives.index(objective)

        if not self._origin_metrics:
            self._origin_metrics = [objective(self._graph).value]
        elif len(self._origin_metrics) <= obj_idx:
            self._origin_metrics.append(objective(self._graph).value)

        modified_graph_metric = objective(modified_graph).value

        if not self._origin_metrics[obj_idx]:
            self.log.warning("Origin graph can not be evaluated")
            return -1.0
        if not modified_graph_metric:
            self.log.warning("Modified graph can not be evaluated")
            return -1.0

        try:
            if modified_graph_metric < 0.0:
                res = modified_graph_metric / self._origin_metrics[obj_idx]
            else:
                res = self._origin_metrics[obj_idx] / modified_graph_metric
        except ZeroDivisionError:
            res = -1.0

        return res


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, graph: OptGraph, objectives: List[Callable],
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objectives, node_factory, requirements)
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: OptNode, **kwargs) -> Dict[str, List[float]]:
        """
        Receives a graph without the specified node and tries to calculate the loss for it

        :param node: OptNode object to analyze
        :return: the ratio of modified graph score to origin score
        """
        if node is self._graph.root_node:
            self.log.warning(f'{node} node can not be deleted')
            return {'loss': [-1.0]*len(self._objectives)}
        else:
            shortened_graph = self.sample(node)
            if shortened_graph:
                losses = self._compare_with_origin_by_metrics(shortened_graph)
                self.log.message(f'losses for {node.name}: {losses}')
                del shortened_graph
            else:
                losses = [-1.0]*len(self._objectives)

            return {'loss': losses}

    def sample(self, node: OptNode):
        """
        Checks if it is possible to delete the node from the graph so that it remains valid,
        and if so, deletes

        :param node: OptNode object to delete from Graph object
        :return: Graph object without node
        """
        graph_sample = deepcopy(self._graph)
        node_index_to_delete = self._graph.nodes.index(node)
        node_to_delete = graph_sample.nodes[node_index_to_delete]

        if node_to_delete.name == 'class_decompose':
            for child in graph_sample.node_children(node_to_delete):
                graph_sample.delete_node(child)

        graph_sample.delete_node(node_to_delete)

        verifier = GraphVerifier()
        if not verifier.verify(graph_sample):
            self.log.message('Can not delete node since modified graph can not be verified')
            return None

        return graph_sample


class NodeReplaceOperationAnalyze(NodeAnalyzeApproach):
    """
    Replace node with operations available for the current task
    and evaluate the score difference
    """

    def __init__(self, graph: OptGraph, objectives: List[Callable],
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objectives, node_factory, requirements)

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: OptNode, **kwargs) -> Dict[str, Union[float, str]]:
        """
        Counts the loss on each changed graph received and returns losses

        :param node: OptNode object to analyze

        :return: the ratio of modified graph score to origin score
        """
        requirements: ReplacementAnalysisMetaParams = self._requirements.replacement_meta
        node_id = self._graph.nodes.index(node)
        samples = self.sample(node=node,
                              nodes_to_replace_to=requirements.nodes_to_replace_to,
                              number_of_random_operations=requirements.number_of_random_operations_nodes)

        try:
            loss_values = []
            new_nodes_types = []
            for sample_graph in samples:
                loss_per_sample = self._compare_with_origin_by_metrics(sample_graph)
                self.log.message(f'losses: {loss_per_sample}\n')
                loss_values.append(loss_per_sample)

                new_node = sample_graph.nodes[node_id]
                new_nodes_types.append(new_node.name)

            loss_and_node_operations = sorted(list(zip(loss_values, new_nodes_types)),
                                              key=lambda x: x[0], reverse=True)
            best_loss = loss_and_node_operations[0][0]
            best_node_operation = loss_and_node_operations[0][1]
        except Exception as ex:
            print(f'HERE9: {ex}')

        return {'loss': best_loss, 'new_node_operation': best_node_operation,
                'all_losses': loss_values, 'node_operations': new_nodes_types}

    def sample(self, node: OptNode,
               nodes_to_replace_to: Optional[List[OptNode]],
               number_of_random_operations: int = 1) -> Union[List[OptGraph], OptGraph]:
        """
        Replaces the given node with a pool of nodes available for replacement (see _node_generation docstring)

        :param node: OptNode object to replace
        :param nodes_to_replace_to: nodes provided for old_node replacement
        :param number_of_random_operations: number of replacement operations, \
        if nodes_to_replace_to not provided
        :return: Sequence of Graph objects with new operations instead of old one
        """

        if not nodes_to_replace_to:
            nodes_to_replace_to = self._node_generation(node=node,
                                                        node_factory=self._node_factory,
                                                        number_of_operations=number_of_random_operations)

        samples = list()
        for replacing_node in nodes_to_replace_to:
            sample_graph = deepcopy(self._graph)
            replaced_node_index = self._graph.nodes.index(node)
            replaced_node = sample_graph.nodes[replaced_node_index]
            sample_graph.update_node(old_node=replaced_node,
                                     new_node=replacing_node)
            verifier = GraphVerifier()
            if not verifier.verify(sample_graph):
                self.log.warning(f'Can not replace {node.name} node with {replacing_node.name} node.')
            else:
                self.log.message(f'replacing node: {replacing_node.name}')
                samples.append(sample_graph)

        if not samples:
            samples.append(self._graph)

        return samples

    @staticmethod
    def _node_generation(node: OptNode,
                         node_factory: OptNodeFactory,
                         number_of_operations: int = 1) -> List[OptNode]:
        """
        The method returns possible nodes that can replace the given node

        :param node: the node to be replaced
        :param number_of_operations: limits the number of possible nodes to replace to

        :return: nodes that can be used to replace
        """

        available_nodes = [node_factory.exchange_node(node=node)]*number_of_operations if number_of_operations \
            else [node_factory.exchange_node(node=node)]

        if number_of_operations:
            available_nodes = [i for i in available_nodes if i != node.name]
            number_of_operations = min(len(available_nodes), number_of_operations)
            random_nodes = random.sample(available_nodes, number_of_operations)
        else:
            random_nodes = available_nodes

        nodes = []
        for node in random_nodes:
            node.nodes_from = node.nodes_from
            nodes.append(node)

        return nodes


class SubtreeDeletionAnalyze(NodeAnalyzeApproach):
    """
    Approach to delete specified node subtree
    """
    def __init__(self, graph: OptGraph, objectives: List[Callable],
                 node_factory: OptNodeFactory,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(graph, objectives, node_factory, requirements)
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') \
            if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: OptNode, **kwargs) -> Dict[str, List[float]]:
        """
        Receives a graph without the specified node's subtree and
        tries to calculate the loss for it

        :param node: OptNode object to analyze
        :return: the ratio of modified graph score to origin score
        """
        if node is self._graph.root_node:
            self.log.warning(f'{node} subtree can not be deleted')
            return {'loss': [-1.0]*len(self._objectives)}
        else:
            shortened_graph = self.sample(node)
            if shortened_graph:
                loss = self._compare_with_origin_by_metrics(shortened_graph)
                self.log.message(f'loss for {node.name}: {loss}')
                del shortened_graph
            else:
                loss = [-1.0]*len(self._objectives)

            return {'loss': loss}

    def sample(self, node: OptNode):
        """
        Checks if it is possible to delete the node's subtree from the graph so that it remains valid,
        and if so, deletes

        :param node: OptNode object from which to delete subtree from OptGraph object
        :return: OptGraph object without subtree
        """
        graph_sample = deepcopy(self._graph)
        node_index_to_delete = self._graph.nodes.index(node)
        node_to_delete = graph_sample.nodes[node_index_to_delete]

        if node_to_delete.name == 'class_decompose':
            for child in graph_sample.node_children(node_to_delete):
                graph_sample.delete_node(child)

        graph_sample.delete_subtree(node_to_delete)

        verifier = GraphVerifier()
        if not verifier.verify(graph_sample):
            self.log.warning('Can not delete subtree since modified graph can not pass verification')
            return None

        return graph_sample
