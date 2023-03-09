import random
from abc import ABC, abstractmethod
from copy import deepcopy
from os import makedirs
from os.path import exists, join
from typing import List, Optional, Type, Union, Dict, Callable, Any

from golem.core.dag.graph_verifier import GraphVerifier
from golem.core.log import default_log
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.timer import OptimisationTimer
from golem.core.paths import default_data_dir
from golem.structural_analysis.pipeline_sa.sa_requirements import StructuralAnalysisRequirements, \
    ReplacementAnalysisMetaParams


class NodeAnalysis:
    """
    :param approaches: methods applied to nodes to modify the pipeline or analyze certain operations.\
    Default: [NodeDeletionAnalyze, NodeTuneAnalyze, NodeReplaceOperationAnalyze]
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, task_type: Any,
                 approaches: Optional[List[Type['NodeAnalyzeApproach']]] = None,
                 approaches_requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):

        self.task_type = task_type

        self.approaches = [NodeDeletionAnalyze, NodeReplaceOperationAnalyze] if approaches is None else approaches

        self.path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        self.log = default_log(self)

        self.approaches_requirements = \
            StructuralAnalysisRequirements() if approaches_requirements is None else approaches_requirements

    def analyze(self, pipeline: OptGraph, node: OptNode,
                objectives: List[Callable],
                timer: OptimisationTimer = None) -> dict:

        """
        Method runs Node analysis within defined approaches

        :param pipeline: Pipeline containing the analyzed Node
        :param node: Node object to analyze in Pipeline
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
                approach(pipeline=pipeline,
                         objectives=objectives,
                         task_type=self.task_type,
                         requirements=self.approaches_requirements,
                         path_to_save=self.path_to_save).analyze(node=node)
        return results

    @staticmethod
    def _get_node_index(train_data: Any, results: dict):
        """
        For each node, give an rating==assessment of "importance" on a scale from 0 to 5
        and display it on the visualization
        """
        total_index = None
        deletion_score = results[NodeDeletionAnalyze.__name__][0]

        if NodeReplaceOperationAnalyze.__name__ in results.keys() and NodeDeletionAnalyze.__name__ in results.keys():
            task = train_data.task.task_type
            app_models, _ = OperationTypesRepository().suitable_operation(task_type=task)
            total_operations_number = len(app_models)

            replacement_candidates = results[NodeReplaceOperationAnalyze.__name__]
            candidates_for_replacement_number = len(
                [candidate for candidate in replacement_candidates if (1 - candidate) < 0])

            replacement_score = candidates_for_replacement_number / total_operations_number

            total_index = (deletion_score / abs(deletion_score)) * replacement_score
        elif NodeDeletionAnalyze.__name__ in results.keys():
            total_index = deletion_score

        return total_index


class NodeAnalyzeApproach(ABC):
    """
    Base class for analysis approach.
    :param pipeline: Pipeline containing the analyzed Node
    :param objectives: objective functions for computing metric values
    :param path_to_save: path to save results to. Default: ~home/Fedot/structural
    """

    def __init__(self, pipeline: OptGraph, objectives: List[Callable],
                 task_type: Any,
                 requirements: StructuralAnalysisRequirements = None,
                 path_to_save=None):
        self._pipeline = pipeline
        self._objectives = objectives
        self._task_type = task_type
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
        """Changes the pipeline according to the approach"""
        pass

    def _is_the_modified_pipeline_different(self, modified_pipeline: OptGraph):
        """ Checks if the pipeline after changes is different from the original pipeline """
        if modified_pipeline.root_node.descriptive_id != self._pipeline.root_node.descriptive_id:
            return True
        return False

    def _compare_with_origin_by_metrics(self, modified_pipeline: OptGraph) -> List[float]:
        """ Iterate through all objectives and evaluate modified pipeline """
        results = []
        for objective in self._objectives:
            metric = self._compare_with_origin_by_metric(modified_pipeline=modified_pipeline,
                                                         objective=objective)
            results.append(metric)
        return results

    def _compare_with_origin_by_metric(self, modified_pipeline: OptGraph,
                                       objective: Callable) -> float:
        """ Returns the ratio of metrics for the modified pipeline and the original one """

        if not self._is_the_modified_pipeline_different(modified_pipeline):
            return -1.0

        obj_idx = self._objectives.index(objective)

        if not self._origin_metrics:
            self._origin_metrics = [objective(self._pipeline).value]
        elif len(self._origin_metrics) <= obj_idx:
            self._origin_metrics.append(objective(self._pipeline).value)

        modified_pipeline_metric = objective(modified_pipeline).value

        if not self._origin_metrics[obj_idx]:
            self.log.warning("Origin pipeline can not be evaluated")
            return -1.0
        if not modified_pipeline_metric:
            self.log.warning("Modified pipeline can not be evaluated")
            return -1.0

        try:
            if modified_pipeline_metric < 0.0:
                res = modified_pipeline_metric / self._origin_metrics[obj_idx]
            else:
                res = self._origin_metrics[obj_idx] / modified_pipeline_metric
        except ZeroDivisionError:
            res = -1.0

        return res


class NodeDeletionAnalyze(NodeAnalyzeApproach):
    def __init__(self, pipeline: OptGraph, objectives: List[Callable],
                 task_type: Any,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(pipeline, objectives, task_type, requirements)
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: OptNode, **kwargs) -> Dict[str, List[float]]:
        """
        Receives a pipeline without the specified node and tries to calculate the loss for it

        :param node: OptNode object to analyze
        :return: the ratio of modified pipeline score to origin score
        """
        if node is self._pipeline.root_node:
            self.log.warning(f'{node} node can not be deleted')
            return {'loss': [-1.0]*len(self._objectives)}
        else:
            shortened_pipeline = self.sample(node)
            if shortened_pipeline:
                losses = self._compare_with_origin_by_metrics(shortened_pipeline)
                self.log.message(f'losses for {node.operation.operation_type}: {losses}')
                del shortened_pipeline
            else:
                losses = [-1.0]*len(self._objectives)

            return {'loss': losses}

    def sample(self, node: OptNode):
        """
        Checks if it is possible to delete the node from the pipeline so that it remains valid,
        and if so, deletes

        :param node: OptNode object to delete from Pipeline object
        :return: Pipeline object without node
        """
        pipeline_sample = deepcopy(self._pipeline)
        node_index_to_delete = self._pipeline.nodes.index(node)
        node_to_delete = pipeline_sample.nodes[node_index_to_delete]

        if node_to_delete.operation.operation_type == 'class_decompose':
            for child in pipeline_sample.node_children(node_to_delete):
                pipeline_sample.delete_node(child)

        pipeline_sample.delete_node(node_to_delete)

        verifier = GraphVerifier()
        if not verifier.verify(pipeline_sample):
            self.log.message('Can not delete node since modified graph can not be verified')
            return None

        return pipeline_sample


class NodeReplaceOperationAnalyze(NodeAnalyzeApproach):
    """
    Replace node with operations available for the current task
    and evaluate the score difference
    """

    def __init__(self, pipeline: OptGraph, objectives: List[Callable],
                 task_type: Any,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(pipeline, objectives, task_type, requirements)

        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: OptNode, **kwargs) -> Dict[str, Union[float, str]]:
        """
        Counts the loss on each changed pipeline received and returns losses

        :param node: OptNode object to analyze

        :return: the ratio of modified pipeline score to origin score
        """
        requirements: ReplacementAnalysisMetaParams = self._requirements.replacement_meta
        node_id = self._pipeline.nodes.index(node)
        samples = self.sample(node=node,
                              nodes_to_replace_to=requirements.nodes_to_replace_to,
                              number_of_random_operations=requirements.number_of_random_operations_nodes)

        try:
            loss_values = []
            new_nodes_types = []
            for sample_pipeline in samples:
                loss_per_sample = self._compare_with_origin_by_metrics(sample_pipeline)
                self.log.message(f'losses: {loss_per_sample}\n')
                loss_values.append(loss_per_sample)

                new_node = sample_pipeline.nodes[node_id]
                new_nodes_types.append(new_node.operation.operation_type)

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
               number_of_random_operations: Optional[int] = None) -> Union[List[OptGraph], OptGraph]:
        """
        Replaces the given node with a pool of nodes available for replacement (see _node_generation docstring)

        :param node: OptNode object to replace
        :param nodes_to_replace_to: nodes provided for old_node replacement
        :param number_of_random_operations: number of replacement operations, \
        if nodes_to_replace_to not provided
        :return: Sequence of Pipeline objects with new operations instead of old one
        """

        if not nodes_to_replace_to:
            nodes_to_replace_to = self._node_generation(node=node,
                                                        task_type=self._task_type,
                                                        number_of_operations=number_of_random_operations)

        samples = list()
        for replacing_node in nodes_to_replace_to:
            sample_pipeline = deepcopy(self._pipeline)
            replaced_node_index = self._pipeline.nodes.index(node)
            replaced_node = sample_pipeline.nodes[replaced_node_index]
            sample_pipeline.update_node(old_node=replaced_node,
                                        new_node=replacing_node)
            verifier = GraphVerifier()
            if not verifier.verify(sample_pipeline):
                self.log.warning(f'Can not replace {node.operation} node with {replacing_node.operation} node.')
            else:
                self.log.message(f'replacing node: {replacing_node.operation}')
                samples.append(sample_pipeline)

        if not samples:
            samples.append(self._pipeline)

        return samples

    @staticmethod
    def _node_generation(node: OptNode,
                         task_type: Any,
                         number_of_operations=None) -> List[OptNode]:
        """
        The method returns possible nodes that can replace the given node

        :param node: the node to be replaced
        :param number_of_operations: limits the number of possible nodes to replace to

        :return: nodes that can be used to replace
        """

        app_operations = OperationTypesRepository().suitable_operation(task_type=task_type)

        if number_of_operations:
            app_operations = [i for i in app_operations if i != node.operation.operation_type]
            number_of_operations = min(len(app_operations), number_of_operations)
            random_operations = random.sample(app_operations, number_of_operations)
        else:
            random_operations = app_operations

        node_type = type(node)
        nodes = []
        for operation in random_operations:
            nodes.append(node_type(operation_type=operation, nodes_from=node.nodes_from))

        return nodes


class SubtreeDeletionAnalyze(NodeAnalyzeApproach):
    """
    Approach to delete specified node subtree
    """
    def __init__(self, pipeline: OptGraph, objectives: List[Callable],
                 task_type: Any,
                 requirements: StructuralAnalysisRequirements = None, path_to_save=None):
        super().__init__(pipeline, objectives, task_type, requirements)
        self._path_to_save = \
            join(default_data_dir(), 'structural', 'nodes_structural') \
                if path_to_save is None else path_to_save
        if not exists(self._path_to_save):
            makedirs(self._path_to_save)

    def analyze(self, node: OptNode, **kwargs) -> Dict[str, List[float]]:
        """
        Receives a pipeline without the specified node's subtree and
        tries to calculate the loss for it

        :param node: OptNode object to analyze
        :return: the ratio of modified pipeline score to origin score
        """
        if node is self._pipeline.root_node:
            self.log.warning(f'{node} subtree can not be deleted')
            return {'loss': [-1.0]*len(self._objectives)}
        else:
            shortened_pipeline = self.sample(node)
            if shortened_pipeline:
                loss = self._compare_with_origin_by_metrics(shortened_pipeline)
                self.log.message(f'loss for {node.operation.operation_type}: {loss}')
                del shortened_pipeline
            else:
                loss = [-1.0]*len(self._objectives)

            return {'loss': loss}

    def sample(self, node: OptNode):
        """
        Checks if it is possible to delete the node's subtree from the pipeline so that it remains valid,
        and if so, deletes

        :param node: OptNode object from which to delete subtree from OptGraph object
        :return: OptGraph object without subtree
        """
        pipeline_sample = deepcopy(self._pipeline)
        node_index_to_delete = self._pipeline.nodes.index(node)
        node_to_delete = pipeline_sample.nodes[node_index_to_delete]

        if node_to_delete.operation.operation_type == 'class_decompose':
            for child in pipeline_sample.node_children(node_to_delete):
                pipeline_sample.delete_node(child)

        pipeline_sample.delete_subtree(node_to_delete)

        verifier = GraphVerifier()
        if not verifier.verify(pipeline_sample):
            self.log.warning('Can not delete subtree since modified graph can not pass verification')
            return None

        return pipeline_sample
