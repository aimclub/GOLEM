from dataclasses import dataclass, replace, field
from enum import Enum
from itertools import chain
from typing import Optional, List, Union, Any, Dict

from golem.core.optimisers.genetic.operators.operator import Operator
from golem.core.optimisers.graph import OptGraph


class TaskStagesEnum(Enum):
    (INIT, SUCCESS, FAIL, FINISH) = range(4)


@dataclass
class GeneticOperatorTask:
    """ Contain graphs and information what to do with it and what was made """
    graphs: List[OptGraph]
    operator_type: Optional[Any] = None

    stage: TaskStagesEnum = TaskStagesEnum.INIT
    stage_node: Optional['GeneticNode'] = None

    # parent data
    parent_task: Optional['GeneticOperatorTask'] = None

    fail: bool = False
    fail_message: str = ''
    left_tries: int = 1

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memodict: Dict = dict()):
        raise NotImplementedError('Deepcopy is not allowed for task')

    def copy(self):
        return replace(self)

    def create_failed_task(self, exception: Exception):
        failed_task = self.copy()
        failed_task.stage = TaskStagesEnum.FAIL
        failed_task.fail_message = exception.__str__()
        failed_task.left_tries -= 1
        return failed_task

    def create_successive_task(self, graphs: List[OptGraph], operator_type: Any):
        successive_task = self.copy()
        successive_task.stage = TaskStagesEnum.SUCCESS
        successive_task.graphs = graphs
        successive_task.operator_type = operator_type
        successive_task.parent_task = self
        return successive_task


@dataclass(frozen=True)
class GeneticNode:
    """ Operator wrapper with data/tools for task routing """

    name: str
    operator: Operator
    success_outputs: List[Union['GeneticNode', None]]
    fail_outputs: Optional[Union[List['GeneticNode'], None]] = field(default_factory=lambda: [None])

    def __post_init__(self):
        # some checks
        _check_list_with_genetic_nodes(self.success_outputs, allow_none=True)
        _check_list_with_genetic_nodes(self.fail_outputs, allow_none=True)

        # TODO check interface of operator

    def __call__(self, task: GeneticOperatorTask):
        if task.left_tries > 0:
            try:
                *grouped_graphs, operator_type = self.operator(task.graphs, task.operator_type)
                tasks = [task.create_successive_task(graphs, operator_type) for graphs in grouped_graphs]
                next_nodes = self.success_outputs
            except Exception as exception:
                tasks = [task.create_failed_task(exception)]
                next_nodes = self.fail_outputs

            final_tasks = list()
            for _task in tasks:
                for _node in next_nodes:
                    new_task = _task.copy()
                    new_task.stage_node = _node.name
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


@dataclass(frozen=True)
class GeneticPipeline:
    """ Pool of connected nodes with useful checks
        Call only a one node in time
    """

    name: str
    nodes: List[GeneticNode]
    __nodes_map: Optional[Dict[int, GeneticNode]] = None

    def __post_init__(self):
        # some checks
        _check_list_with_genetic_nodes(self.nodes)

        # check that all connection between nodes connect existing nodes
        connection_goals = set(chain(*[chain(*(node.success_outputs + node.fail_outputs)) for node in self.nodes]))
        if not (set(self.nodes) > connection_goals):
            raise ValueError('Some nodes have connection with nonexisting nodes')

        self.__setattr__('__nodes_map', {node: node for node in self.nodes})

        if self.__nodes_map is None:
            raise ValueError('there is no ``__nodes_map``')

    def __call__(self, task: GeneticOperatorTask):
        """ Call one of node and return result """
        if not isinstance(task, GeneticOperatorTask):
            raise ValueError(f"``task`` should be ``GeneticOperatorTask``, get {type(task)} instead")

        if task.stage in (TaskStagesEnum.INIT, TaskStagesEnum.FINISH):
            raise ValueError('Unappropriate task')

        if task.stage_node not in self.__nodes_map:
            raise ValueError(f"Unknown stage node {task.stage}")

        return self.__nodes_map[task.stage_node](task)


def _check_list_with_genetic_nodes(list_with_nodes, allow_none=False):
    # check that nodes is list with nodes
    list_with_nodes_is_appropriate = True
    list_with_nodes_is_appropriate &= isinstance(list_with_nodes, list)
    list_with_nodes_is_appropriate &= len(list_with_nodes) > 0
    if allow_none:
        list_with_nodes_is_appropriate &= all(isinstance(node, GeneticNode) or node is None for node in list_with_nodes)
    else:
        list_with_nodes_is_appropriate &= all(isinstance(node, GeneticNode) for node in list_with_nodes)

    if not list_with_nodes_is_appropriate:
        raise ValueError('``nodes`` parameter should be list with ``GeneticNodes``')

    # check that all nodes have unique name
    # hash of node is calculated as hash of it is name, therefore check may be done as:
    if len(set(list_with_nodes)) != len(list_with_nodes):
        # TODO add test for that line
        # TODO add test for that line works as is
        raise AttributeError(f"nodes names should be unique")
