from dataclasses import dataclass, replace, field
from enum import Enum
from itertools import chain
from math import ceil
from typing import Optional, List, Union, Any, Dict

from golem.core.optimisers.genetic.operators.operator import Operator
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.opt_history_objects.individual import Individual

GeneticNodeAllowedType = Union['GeneticNode', str, None]


class TaskStagesEnum(Enum):
    (INIT, SUCCESS, FAIL, FINISH) = range(4)


@dataclass
class GeneticOperatorTask:
    """ Contain individuals and information what to do with it and what was made """
    individuals: List[Individual]
    operator_type: Optional[Any] = None

    stage: TaskStagesEnum = TaskStagesEnum.INIT
    next_stage_node: GeneticNodeAllowedType = None
    prev_stage_node: GeneticNodeAllowedType = None

    # parent data
    parent_task: Optional['GeneticOperatorTask'] = None

    exception: Optional[Exception] = None
    left_tries: int = 1

    def __repr__(self):
        s = (f"{self.__class__.__name__}('{self.stage.name}', "
             f"next: '{self.next_stage_node}', prev: '{self.prev_stage_node}', "
             f"individuals: {len(self.individuals) if isinstance(self.individuals, list) else type(self.individuals)}, "
             f"operator_type: '{self.operator_type}', "
             f"tries: {self.left_tries}, "
             f"parent: {int(self.parent_task is not None)})")
        return s

    def __copy__(self):
        # TODO test
        return self.copy()

    def __deepcopy__(self, memodict: Dict = dict()):
        # TODO test
        raise NotImplementedError('Deepcopy is not allowed for task')

    def copy(self, **parameters):
        # TODO test
        new_task = replace(self)
        for parameter, value in parameters.items():
            setattr(new_task, parameter, value)
        return new_task

    def create_failed_task(self, exception: Exception, **parameters):
        parameters = {'stage': TaskStagesEnum.FAIL, 'exception': exception,
                      'left_tries': self.left_tries - 1, **parameters}
        return self.copy(**parameters)

    def create_successive_task(self, individuals: List[Individual], **parameters):
        if not isinstance(individuals, list):
            raise ValueError(f"individuals should be list, got {type(individuals)} instead")
        parameters = {'stage': TaskStagesEnum.SUCCESS, 'individuals': individuals,
                      'parent_task': self, **parameters}
        return self.copy(**parameters)


@dataclass(frozen=True)
class GeneticNode:
    """ Operator wrapper with data/tools for task routing """

    name: str
    operator: Operator
    success_outputs: Optional[List[GeneticNodeAllowedType]] = field(default_factory=lambda: [None])
    fail_outputs: Optional[List[GeneticNodeAllowedType]] = field(default_factory=lambda: [None])

    task_params_if_success: Dict[str, Any] = field(default_factory=dict)
    task_params_if_fail: Dict[str, Any] = field(default_factory=dict)

    individuals_input_count: Optional[int] = None
    repeat_count: int = 1
    tries_count: int = 1

    def __post_init__(self):
        # some checks
        _check_list_with_genetic_nodes(self.success_outputs)
        _check_list_with_genetic_nodes(self.fail_outputs)

        # TODO check interface of operator

    def __call__(self, task: GeneticOperatorTask):
        final_tasks = list()

        if task.stage is not TaskStagesEnum.FAIL:
            # if task from previous node then set max tries
            task.left_tries = self.tries_count

            # if there are unappropriated individuals count
            # then divide task to subtasks with appropriate individuals count
            length, max_length = len(task.individuals), self.individuals_input_count
            if max_length is not None and length > max_length:
                individuals_groups = [task.individuals[i * max_length:min(length, (i + 1) * max_length)]
                                      for i in range(ceil(length / max_length))]
                for individuals_group in individuals_groups:
                    final_tasks.append(task.copy(individuals=individuals_group))
                # get task for current run
                task = final_tasks.pop()

            # repeat each task if it is allowed
            if self.repeat_count > 1:
                final_tasks.append(task)
                for _ in range(self.repeat_count - 1):
                    final_tasks.extend([task.copy() for task in final_tasks])
                # get task for current run
                task = final_tasks.pop()

        # run operator
        if task.stage is not TaskStagesEnum.FAIL or task.left_tries > 0:
            try:
                # TODO all operator should return list of lists of graph
                individuals, operator_type = self.operator(task.individuals, task.operator_type)
                tasks = [task.create_successive_task(individuals, prev_stage_node=self.name,
                                                     operator_type=None, **self.task_params_if_success)]
                next_nodes = self.success_outputs
            except Exception as exception:
                # TODO save where it fails
                tasks = [task.create_failed_task(exception, **self.task_params_if_fail)]
                next_nodes = self.fail_outputs

            for _task in tasks:
                for _node in next_nodes:
                    new_task = _task.copy()
                    if _node is None:
                        if new_task.stage is TaskStagesEnum.SUCCESS:
                            new_task.stage = TaskStagesEnum.FINISH
                        elif new_task.stage is TaskStagesEnum.FAIL:
                            # if there is no next node, then no tries
                            new_task.left_tries = -1
                    new_task.next_stage_node = _node
                    final_tasks.append(new_task)
        return final_tasks

    def __hash__(self):
        # TODO add test for hash
        return self.name.__hash__()

    def __copy__(self):
        """ because hash is the name """
        raise NotImplementedError('Use ``copy`` function instead')

    def __deepcopy__(self, memodict: Dict = dict()):
        """ because hash is the name """
        raise NotImplementedError('Use ``copy`` function instead')

    def copy(self, name: str):
        """ Create new node with same data but new name """
        # TODO add tests that all fields are copied
        # new_node = replace(self)
        return GeneticNode(name=name, operator=self.operator,
                               success_outputs=self.success_outputs,
                               fail_outputs=self.fail_outputs)

    # def call_operation(self, task: GeneticOperatorTask):
    #     graphs_grouped, operator_type = self.operator(task.graphs, task.operator_type)
    #     graphs_grouped = [([graph] if not isinstance(graph, list) else graph) for graph in graphs_grouped]
    #
    #     new_graphs_grouped = list()
    #     for graphs in graphs_grouped:
    #         if len(graphs) > self.max_graphs_output:
    #             raise NotImplementedError()
    #         else:
    #             new_graphs_grouped.append(graphs)
    #     return graphs, operator_type


@dataclass
class GeneticPipeline:
    """ Pool of connected nodes with useful checks
        Call only a one node in time
    """

    name: str
    nodes: List[GeneticNode]
    __nodes_map: Optional[Dict[str, GeneticNode]] = None

    def __post_init__(self):
        # some checks
        _check_list_with_genetic_nodes(self.nodes, force_genetic_node_type_check=True)

        # check that all connection between nodes connect existing nodes
        # TODO fix
        # connection_goals = set(chain(*[node.success_outputs + node.fail_outputs for node in self.nodes]))
        # connection_goals -= {None}
        # if not (set(self.nodes) > connection_goals):
        #     raise ValueError('Some nodes have connection with nonexisting nodes')

        self.__nodes_map = {node.name: node for node in self.nodes}

    def __call__(self, task: GeneticOperatorTask):
        """ Call one node and return result """
        if not isinstance(task, GeneticOperatorTask):
            raise ValueError(f"``task`` should be ``GeneticOperatorTask``, get {type(task)} instead")

        if task.stage is TaskStagesEnum.FINISH:
            raise ValueError('Task is finished')

        if task.next_stage_node not in self.__nodes_map:
            raise ValueError(f"Unknown stage node {task.stage}")

        return self.__nodes_map[task.next_stage_node](task)

    def __getitem__(self, node_name: str):
        if node_name not in self.__nodes_map:
            raise KeyError(f"Unknown node {node_name}")
        return self.__nodes_map[node_name]

    def __contains__(self, node_name: str):
        # TODO test that contains also return true when getitem works
        return node_name in self.__nodes_map

def _check_list_with_genetic_nodes(list_with_nodes, force_genetic_node_type_check=False):
    # check that nodes is list with nodes
    list_with_nodes_is_appropriate = True
    list_with_nodes_is_appropriate &= isinstance(list_with_nodes, list)
    list_with_nodes_is_appropriate &= len(list_with_nodes) > 0
    checked_type = GeneticNode if force_genetic_node_type_check else GeneticNodeAllowedType
    # TODO fix it
    # list_with_nodes_is_appropriate &= all(isinstance(node, checked_type) for node in list_with_nodes)

    if not list_with_nodes_is_appropriate:
        raise ValueError('``nodes`` parameter should be list with ``GeneticNodes``')

    # check that all nodes have unique name
    # hash of node is calculated as hash of it is name, therefore check may be done as:
    if len(set(list_with_nodes)) != len(list_with_nodes):
        # TODO add test for that line
        # TODO add test for that line works as is
        raise AttributeError(f"nodes names should be unique")
