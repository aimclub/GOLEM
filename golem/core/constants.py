import numpy as np

MAX_GRAPH_GEN_ATTEMPTS = 1000
MAX_TUNING_METRIC_VALUE = np.inf
MIN_TIME_FOR_TUNING_IN_SEC = 3
# Max number of evaluations attempts to collect the next pop; See usages.
EVALUATION_ATTEMPTS_NUMBER = 10
# Min pop size to avoid getting stuck in local maximum during optimization.
MIN_POP_SIZE = 5
DEFAULT_API_TIMEOUT_MINUTES = 5.0
