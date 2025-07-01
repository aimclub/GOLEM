from functools import partial
from typing import Callable


def get_callable_name(action: Callable):
    if isinstance(action, str):
        return action
    if isinstance(action, partial):
        return action.func.__name__
    try:
        return action.__name__
    except AttributeError:
        return str(action)
