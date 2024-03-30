from typing import Optional

import numpy as np
import random

from golem.utilities.random import RandomStateHandler


def set_random_seed(seed: Optional[int]):
    """ Sets random seed for evaluation of models. """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        RandomStateHandler.MODEL_FITTING_SEED = seed
