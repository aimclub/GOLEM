from collections import defaultdict
from typing import List, Union, Optional

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task
from golem.utilities.utilities import determine_n_jobs


class Scheme:
    def __init__(self, n_jobs: int = -1):
        self.__map = dict()
        self.n_jobs = determine_n_jobs(n_jobs)

    def next(self, task: Task):
        task = task.copy()
        task.set_next_node(self.__map[task.status[0]][task.status[1]])
        return task


class SequentialScheme(Scheme):
    def __init__(self, *args, nodes: Optional[List[Union[str, Node]]] = None, **kwargs):
        if nodes is None:
            raise ValueError('nodes should be list with nodes')
        super().__init__(*args, **kwargs)

        self.__map = dict()
        nodes = [node.name if isinstance(node, Node) else node for node in nodes]
        for prev_node, next_node in zip([None] + nodes, nodes + [None]):
            self.__map[prev_node] = defaultdict(next_node)
