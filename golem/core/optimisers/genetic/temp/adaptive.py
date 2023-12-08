from golem.core.optimisers.common_optimizer.common_optimizer import CommonOptimizerParameters
from golem.core.optimisers.common_optimizer.task import Task
from golem.core.optimisers.optimizer import OptimizationParameters, GraphGenerationParams, AlgorithmParameters


class AdaptiveParametersTask(Task):
    """
    This class is a Memento for storing a state of OptimizationParameters,
    GraphGenerationParams and AlgorithmParameters in CommonOptimizerParameters.
    :param parameters: instance of CommonOptimizerParameters containing the initial parameters
    """

    def __init__(self, parameters: CommonOptimizerParameters):
        super().__init__()
        self.parameters = {}
        for attribute, params_values in parameters.__dict__.items():
            if isinstance(params_values, (OptimizationParameters, GraphGenerationParams, AlgorithmParameters)):
                parameters[attribute] = dict(params_values.__dict__.items())

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
