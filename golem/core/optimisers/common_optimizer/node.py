from dataclasses import dataclass
from typing import Callable, Union, List

from golem.core.optimisers.common_optimizer.task import Task


@dataclass
class Node:
    """ Node with operation """

    name: str
    operation: Callable[[Task], Union[Task, List[Task]]]

    def __call__(self, *args, **kwargs):
        results = self.operation(*args, **kwargs)
        if not isinstance(results, list):
            results = [results]
        if all(isinstance(result, Task) for result in results):
            return results
        raise ValueError('results should be list with tasks')
