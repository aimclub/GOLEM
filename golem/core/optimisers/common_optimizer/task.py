from copy import deepcopy
from enum import Enum, auto

from typing import Optional, Tuple, List, Any


class TaskStatusEnum(Enum):
    NEXT = auto()
    SUCCESS = auto()
    FAIL = auto()
    FINISH = auto()


class Task:
    def __init__(self, parameters: Optional[Any] = None):
        self._stages: List[Tuple[Optional[str], TaskStatusEnum]] = [(None, TaskStatusEnum.NEXT)]

    def __repr__(self):
        params = [f"{name}={val}"for name, val in self.__dict__.items()
                  if not name.startswith('_')]
        return (f"{self.__class__.__name__}({', '.join(params)})"
                f"(status={self.status.name},node={self.node},history={len(self._stages)})")

    @property
    def status(self):
        return self._stages[-1][-1]

    @status.setter
    def status(self, item: TaskStatusEnum):
        if not isinstance(item, TaskStatusEnum):
            raise TypeError(f"status should be `TaskStatusEnum`, got {type(item)} instead")
        self._stages.append((self.node, item))

    @property
    def node(self):
        return self._stages[-1][0]

    @node.setter
    def node(self, item: Optional[str]):
        if not isinstance(item, str) and item is not None:
            raise TypeError(f"node should be `str` or `None`, got {type(item)} instead")
        self._stages.append((item, TaskStatusEnum.FINISH if item is None else TaskStatusEnum.NEXT))

    def copy(self):
        return deepcopy(self)
