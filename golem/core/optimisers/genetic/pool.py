from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Any, Callable


class ParametersTypesEnum(Enum):
    UNKNOWN = auto()
    OPTIMIZER = auto()
    POOL = auto()
    NODE = auto()

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __next__(self):
        return ParametersTypesEnum(self.value + 1)


# class Parameters:
#     def __init__(self, type_: ParametersTypesEnum, data: Optional[dict] = None):
#         data = data or dict()
#
#         for k in data:
#             if isinstance(data[k], dict):
#                 data[k] = Parameters(next(type_), data[k])
#         self.type = type_
#         self.__data = data
#
#     def __getitem__(self, keys):
#         data = self.__data
#         for key in keys:
#             data = data[key]
#         return data
#
#     def __setitem__(self, keys, value):
#         data = self.__data
#         for key in keys[:-1]:
#             if key not in data:
#                 data[key] = Parameters(next(self.type))
#             data = data[key]
#         data[keys[-1]] = value
#
#     def __repr__(self):
#         def pp(parameters, indent=0):
#             return '\n' + '\n'.join(f"{' ' * indent}'{key}': {value.type.name + pp(value, indent + 2) if isinstance(value, self.__class__) else value}"
#                              for key, value in parameters.__data.items())
#         return self.type.name + pp(self)
#
#     def __iter__(self):
#         return (x for x in self.__data.keys())
#
#     def items(self):
#         return (x for x in self.__data.items())
#
#     def filter_by_type(self, type_: ParametersTypesEnum):
#         return [pars for name, pars in self.items()
#                 if isinstance(pars, Parameters) and pars.type is type_]


class Parameters:
    pass


@dataclass
class OptimizerParameters(Parameters):
    pool_parameters: List['PoolParameters']
    n_jobs: int = -1


@dataclass
class PoolParameters(Parameters):
    name: str
    constructor: Callable
    n_jobs: int
    nodes: List['Node']
    scheme: 'Scheme'
    task_constructor: Callable
    task_history: List[Any]


class Optimizer:
    def __init__(self, parameters: OptimizerParameters):
        self.parameters = parameters

    def _evolve_population(self):
        common_parameters = self.parameters
        for pool_params in common_parameters.pool_parameters:
            pool = pool_params.constructor(pool_params, common_parameters)
            common_parameters.update(pool.run())


class Pool:
    """ Pool of nodes """

    def __init__(self, pool_parameters: PoolParameters, parameters: OptimizerParameters):
        self.name = pool_parameters.name
        self.nodes_map = {node.name: node for node in pool_parameters.nodes}
        self.task = pool_parameters.task
        self.scheme = pool_parameters.scheme

        # TODO error if there are some nodes with same name

    def __call__(self, task: Task):
        if not task.next in self.nodes_map:
            raise ValueError((f"Pool {self.name}. Unknown node {task.next}. "
                              f"Existing nodes: {', '.join(self.nodes_map)}."))
        processed_task = task.run_on_node(self.nodes_map[task.next])
        return processed_task


class Node:
    """ Node with operation """

    def __init__(self, name: str, operation: Callable):
        self.name = name
        self.operation = operation

    def __call__(self, *args, **kwargs):
        return self.operation(*args, **kwargs)


class Task:
    """ Data with parameters for operation """

    def __init__(self, data: Any, parameters: Any):
        self.data = data
        self.parameters = parameters

    def run_on_node(self, node: Node):
        result = node(self.data, self.parameters)
