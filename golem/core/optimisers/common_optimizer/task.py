from copy import deepcopy
from enum import Enum, auto

from typing import Optional, Tuple, List, Any


class TaskStatusEnum(Enum):
    NEXT = auto()
    SUCCESS = auto()
    FAIL = auto()
    FINISH = auto()


class TaskMixin:
    """
    Base class of task.
    """
    def update_parameters(self, parameters: 'CommonOptimizerParameters'):
        return parameters

    def copy(self):
        return deepcopy(self)



class Task(TaskMixin):
    """
    Provides functionality to extract, contain, and inject data from/to CommonOptimizerParameters.
    Task is used as a data container for scheme data streams and is essentially a wrapper for the data.

    Methods:
        __init__(parameters: CommonOptimizerParameters):
            Extracts data from the CommonOptimizerParameters object.

        update_parameters(task: Task):
            Injects parameters from the task into the CommonOptimizerParameters object.
    """

    def __init__(self, parameters: Optional['CommonOptimizerParameters'] = None):
        # history of scheme data flows is stored in `self._stages`
        self._stages: List[Tuple[Optional[str], TaskStatusEnum]] = [(None, TaskStatusEnum.NEXT)]

    def __repr__(self):
        params = [f"{name}={val}"for name, val in self.__dict__.items()
                  if not name.startswith('_')]
        return (f"{self.__class__.__name__}({', '.join(params)})"
                f"(status={self.status.name},node={self.node},history={len(self._stages)})")

    @property
    def status(self):
        """ Retrieve task status from current last stage of task """
        return self._stages[-1][-1]

    @status.setter
    def status(self, item: TaskStatusEnum):
        """ Set specific task status to current last stage of task """
        if not isinstance(item, TaskStatusEnum):
            raise TypeError(f"status should be `TaskStatusEnum`, got {type(item)} instead")
        self._stages.append((self.node, item))

    @property
    def node(self):
        """ Retrieve current node name from current last stage of task """
        return self._stages[-1][0]

    @node.setter
    def node(self, item: Optional[str]):
        """ Set specific node name to current last stage of task """
        if not isinstance(item, str) and item is not None:
            raise TypeError(f"node should be `str` or `None`, got {type(item)} instead")
        self._stages.append((item, TaskStatusEnum.FINISH if item is None else TaskStatusEnum.NEXT))
