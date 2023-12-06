from copy import deepcopy
from enum import Enum, auto

from typing import Optional, Tuple, List


class TaskStatusEnum(Enum):
    NEXT = auto()
    SUCCESS = auto()
    FAIL = auto()
    FINISH = auto()
    FINISH_RUNNER = auto()


class Task:
    def __init__(self):
        self.__stages: List[Tuple[Optional[str], TaskStatusEnum]] = [(None, TaskStatusEnum.NEXT)]

    @property
    def status(self):
        return self.__stages[-1][-1]

    def set_next_node(self, next_node: Optional[str] = None):
        status = TaskStatusEnum.FINISH if next_node is None else TaskStatusEnum.NEXT
        self.__stages.append((next_node, status))

    def get_next_node(self):
        self.__stages[-1][0]

    def copy(self):
        return deepcopy(self)
