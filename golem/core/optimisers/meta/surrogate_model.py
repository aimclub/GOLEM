from abc import abstractmethod
from typing import Any

import numpy as np

from golem.core.dag.graph import Graph


class SurrogateModel:
    @abstractmethod
    def __call__(self, graph: Graph, **kwargs: Any):
        raise NotImplementedError()


class RandomValuesSurrogateModel(SurrogateModel):
    def __call__(self, graph: Graph, **kwargs: Any):
        return np.random.random(1)
