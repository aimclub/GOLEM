import os
import platform
import tempfile

from pathlib import Path
from typing import Callable


def copy_doc(source_func: Callable) -> Callable:
    """
    Copies a docstring from the provided ``source_func`` to the wrapped function

    :param source_func: function to copy the docstring from

    :return: wrapped function with the same docstring as in the given ``source_func``
    """
    def wrapper(func: Callable) -> Callable:
        func.__doc__ = source_func.__doc__
        return func
    return wrapper


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent.parent


def default_fedot_data_dir() -> str:
    """ Returns the folder where all the output data
    is recorded to. Default: home/Fedot
    """
    temp_folder = Path("/tmp" if platform.system() == "Darwin" else tempfile.gettempdir())
    default_data_path = os.path.join(temp_folder, 'FEDOT')

    if 'FEDOT' not in os.listdir(temp_folder):
        os.mkdir(default_data_path)

    return default_data_path
