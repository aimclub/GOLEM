from dataclasses import dataclass
from typing import Callable, Union, List

from golem.core.optimisers.common_optimizer.task import Task


@dataclass
class Node:
    """
    Represents a lightweight node in a computational graph.
    Implemented to support an adaptive approach to task completion.

    Args:
        name: node identificator
        operation: action performed within the node call
    """

    def __init__(self, name: str, operation: Callable[[Task], Union[Task, List[Task]]]):
        self.name = name
        self.operation = operation

    def __call__(self, *args, **kwargs):
        results = self.operation(*args, **kwargs)
        if not isinstance(results, list):
            results = [results]
        if all(isinstance(result, Task) for result in results):
            return results
        raise ValueError('results should be list with tasks')
