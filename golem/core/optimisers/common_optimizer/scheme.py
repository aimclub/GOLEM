from collections import defaultdict
from typing import List, Union, Optional, Dict

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task, TaskStatusEnum
from golem.utilities.utilities import determine_n_jobs


class Scheme:
    """ Contain pipeline for task flow between nodes """
    # TODO create real pipelines with `show` method
    # TODO support for multioutput

    def __init__(self, scheme_map: Optional[Dict[str, Dict[TaskStatusEnum, str]]] = None):
        self._map = scheme_map or dict()
        self.nodes = None

    def next(self, task: Task):
        task.node = self._map[task.node][task.status]
        return task


class SequentialScheme(Scheme):
    def __init__(self, *args, nodes: Optional[List[Union[str, Node]]] = None, **kwargs):
        if nodes is None:
            raise ValueError('nodes should be list with nodes')
        super().__init__(*args, **kwargs)

        self._map = dict()
        nodes = [node.name if isinstance(node, Node) else node for node in nodes]
        for prev_node, next_node in zip([None] + nodes, nodes + [None]):
            self._map[prev_node] = defaultdict(lambda *args, next_node=next_node: next_node)
