from typing import Any, Dict, List

from golem.core.optimisers.common_optimizer.common_optimizer import CommonOptimizerParameters
from golem.core.optimisers.common_optimizer.node import Node
from golem.core.optimisers.common_optimizer.task import Task
from golem.core.optimisers.optimizer import OptimizationParameters, GraphGenerationParams, AlgorithmParameters


class AdaptiveParametersTask(Task):
    """
    This class is for storing a state of OptimizationParameters,
    GraphGenerationParams and AlgorithmParameters in CommonOptimizerParameters.
    :param parameters: instance of CommonOptimizerParameters containing the initial parameters
    """

    def __init__(self, parameters: CommonOptimizerParameters):
        super().__init__()
        self.parameters = {}
        for attribute, values in parameters.__dict__.items():
            if isinstance(values, (OptimizationParameters, GraphGenerationParams, AlgorithmParameters)):
                self.parameters[attribute] = dict(values.__dict__.items())

    def update_parameters(self, parameters: CommonOptimizerParameters) -> CommonOptimizerParameters:
        """
        Update the parameters in CommonOptimizerParameters with stored
        OptimizationParameters, GraphGenerationParams and AlgorithmParameters values.
        :param parameters: instance of CommonOptimizerParameters to update
        :return: updated parameters object
        """
        for attribute, values in self.parameters.items():
            parameters_obj = getattr(parameters, attribute, None)
            if parameters_obj:
                for subattribute, subvalues in values.items():
                    setattr(parameters_obj, subattribute, subvalues)
        return parameters


class AdaptiveParameters(Node):
    def __init__(self, name: str, parameters: Dict[str, Dict[str, Any]]):
        self.name = name
        self.parameters = parameters

    def update_parameters(self, task: AdaptiveParametersTask) -> List[AdaptiveParametersTask]:
        if not isinstance(task, AdaptiveParametersTask):
            raise TypeError(f"task should be `AdaptiveParametersTask`, got {type(task)} instead")
        for attribute, values in self.parameters:
            parameters_dict = task.parameters.get(attribute, None)
            if parameters_dict:
                for subattribute, subvalues in values.items():
                    if subattribute in parameters_dict:
                        parameters_dict[subattribute] = subvalues
        return [task]