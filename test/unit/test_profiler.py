import os
import shutil
import datetime

import pytest

from examples.graph_model_optimization import run_custom_example
from golem.utilities.profiler.memory_profiler import MemoryProfiler
from golem.utilities.profiler.time_profiler import TimeProfiler


@pytest.fixture(scope='session', autouse=True)
def preprocessing_files_before_and_after_tests(request):
    path = ['time_profiler', 'memory_profiler']

    delete_files = create_func_delete_files(path)
    request.addfinalizer(delete_files)


def create_func_delete_files(paths):
    """
    Create function to delete files that created after tests.
    """

    def wrapper():
        for path in paths:
            if os.path.isdir(path) or path.endswith('.log'):
                shutil.rmtree(path)

    return wrapper


def test_time_profiler_correctly():
    """
    Profilers requirements are needed
    """
    profiler = TimeProfiler()
    run_custom_example(timeout=datetime.timedelta(minutes=0.5), visualisation=False)
    path = os.path.abspath('time_profiler')
    profiler.profile(path=path, node_percent=0.5, edge_percent=0.1, open_web=False)

    assert os.path.exists(path)


def test_memory_profiler_correctly():
    """
    Profilers requirements are needed
    """

    arguments = dict(timeout=datetime.timedelta(minutes=0.5), visualisation=False)
    path = os.path.abspath('time_profiler')
    MemoryProfiler(run_custom_example, kwargs=arguments,
                   path=path, roots=[run_custom_example], max_depth=8)

    assert os.path.exists(path)
