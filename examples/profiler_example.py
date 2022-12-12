import os
import random
import datetime

import numpy as np

from examples.graph_model_optimization import run_custom_example
from golem.utilities.profiler.memory_profiler import MemoryProfiler
from golem.utilities.profiler.time_profiler import TimeProfiler

random.seed(1)
np.random.seed(1)

if __name__ == '__main__':
    # JUST UNCOMMENT WHAT TYPE OF PROFILER DO YOU NEED
    # EXAMPLE of MemoryProfiler.

    arguments = dict(timeout=datetime.timedelta(minutes=0.5), visualisation=False)
    path = os.path.join(os.path.expanduser("~"), 'memory_profiler')
    MemoryProfiler(run_custom_example, kwargs=arguments,
                   path=path, roots=[run_custom_example], max_depth=8, visualization=True)

    # EXAMPLE of TimeProfiler.

    profiler = TimeProfiler()
    run_custom_example(timeout=datetime.timedelta(minutes=0.5), visualisation=False)
    path = os.path.join(os.path.expanduser("~"), 'time_profiler')
    profiler.profile(path=path, node_percent=0.5, edge_percent=0.1, open_web=True)
