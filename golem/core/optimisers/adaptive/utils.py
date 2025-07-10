from functools import partial
from typing import Callable, Union


def get_callable_name(action: Union[Callable, str]) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, partial):
        return action.func.__name__
    try:
        return action.__name__
    except AttributeError:
        return str(action)
