from typing import Callable

from golem.core.optimisers.common_optimizer.task import Task


class Node:
    """ Node with operation """

    def __init__(self, name: str, operation: Callable[[Task], Task]):
        self.name = name
        self.operation = operation

    def __call__(self, *args, **kwargs):
        results = self.operation(*args, **kwargs)
        if not isinstance(results, list):
            results = [results]
        if all(isinstance(result, Task) for result in results):
            return results
        raise ValueError('results should be list with tasks')
