from typing import Any, Dict, List

from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task
from golem.core.optimisers.optimizer import OptimizationParameters, GraphGenerationParams, AlgorithmParameters
from golem.core.optimisers.genetic.parameters.population_size import AdaptivePopulationSize as OldAdaptivePopulationSize
from golem.core.optimisers.genetic.parameters.population_size import init_adaptive_pop_size


class AdaptiveParametersTask(Task):
    """
    This class is for storing a state of OptimizationParameters,
    GraphGenerationParams and AlgorithmParameters in CommonOptimizerParameters.
    :param parameters: instance of CommonOptimizerParameters containing the initial parameters
    """

    def __init__(self, parameters: 'CommonOptimizerParameters'):
        super().__init__()
        self.graph_optimizer_params = parameters.graph_optimizer_params
        self.graph_generation_params = parameters.graph_generation_params
        self.population = parameters.population

    def update_parameters(self, parameters: 'CommonOptimizerParameters') -> 'CommonOptimizerParameters':
        """
        Update the parameters in CommonOptimizerParameters with stored
        OptimizationParameters, GraphGenerationParams and AlgorithmParameters values.
        :param parameters: instance of CommonOptimizerParameters to update
        :return: updated parameters object
        """
        parameters.graph_generation_params = self.graph_generation_params
        return parameters


class AdaptiveParameters(Node):
    """
    This class is a field-setter for a list of AdaptiveParametersTask,
    new parameters should be passed in a form of double mested dictionaary with
    OptimizationParameters, GraphGenerationParams or AlgorithmParameters specification.
    :param parameters: dictionary with specified parameters and their values
    """
    def __init__(self, name: str, parameters: Dict[str, Dict[str, Any]]):
        self.name = name
        self.parameters = parameters

    def __call__(self, task: AdaptiveParametersTask) -> List[AdaptiveParametersTask]:
        """
        Set the parameters in AdaptiveParametersTask state.
        :param parameters: instance of AdaptiveParametersTask task to set new parameters
        :return: updated AdaptiveParametersTask wrapped in a list
        """
        if not isinstance(task, AdaptiveParametersTask):
            raise TypeError(f"task should be `AdaptiveParametersTask`, got {type(task)} instead")
        for attribute, values in self.parameters:
            parameters_dict = task.parameters.get(attribute, None)
            if parameters_dict:
                for subattribute, subvalues in values.items():
                    if subattribute in parameters_dict:
                        parameters_dict[subattribute] = subvalues
        return [task]


class AdaptivePopulationSize(Node):
    def __init__(self, name: str = 'pop_size'):
        self.name = name
        self._pop_size = None

    def __call__(self, task: AdaptiveParametersTask) -> List[AdaptiveParametersTask]:
        if not isinstance(task, AdaptiveParametersTask):
            raise TypeError(f"task should be `AdaptiveParametersTask`, got {type(task)} instead")
        if self._pop_size is None:
            self._pop_size: OldAdaptivePopulationSize = init_adaptive_pop_size(
                    task.graph_optimizer_params,
                    task.population
                )
        pop_size = self._pop_size.next(task.population)

        task.graph_generation_params.pop_size = pop_size
        return [task]
