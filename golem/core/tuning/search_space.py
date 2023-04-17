from typing import Dict, Callable, List, Union

import numpy as np
from hyperopt import hp


class SearchSpace:
    """
    Args:
        search_space: dictionary with parameters and their search_space
            {'operation_name': {'param_name': {'hyperopt-dist': hyperopt distribution function,
            'sampling-scope': [sampling scope], 'type': 'discrete' or 'continuous'}, ...}, ...},
            e.g. ``{'operation_name': {'param1': {'hyperopt-dist': hp.uniformint, 'sampling-scope': [2, 21]),
            'type': 'discrete'}, ...}, ..}
    """

    def __init__(self, search_space: Dict[str, Dict[str, Dict[str, Union[Callable, List, str]]]]):
        self.parameters_per_operation = search_space

    def get_node_parameters_for_iopt(self, node_id, operation_name):
        """
        Method for forming dictionary with hyperparameters of node operation for the ``IOptTuner``

        Args:
            node_id: number of node in graph.nodes list
            operation_name: name of operation in the node

        Returns:
            float_parameters_dict: dictionary-like structure with labeled float hyperparameters
            and their range per operation
            discrete_parameters_dict: dictionary-like structure with labeled discrete hyperparameters
            and their range per operation
        """
        # Get available parameters for operation
        parameters_dict = self.parameters_per_operation.get(operation_name)

        discrete_parameters_dict = {}
        float_parameters_dict = {}

        if parameters_dict is not None:

            for parameter_name, parameter_properties in parameters_dict.items():
                node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

                parameter_type = parameter_properties.get('type')
                if parameter_type == 'discrete':
                    discrete_parameters_dict.update({node_op_parameter_name: parameter_properties
                                                    .get('sampling-scope')})
                elif parameter_type == 'continuous':
                    float_parameters_dict.update({node_op_parameter_name: parameter_properties
                                                 .get('sampling-scope')})

        return float_parameters_dict, discrete_parameters_dict

    def get_parameters_for_operation(self, operation_name: str) -> List[str]:
        parameters_list = list(self.parameters_per_operation.get(operation_name, {}).keys())
        return parameters_list


def get_node_operation_parameter_label(node_id: int, operation_name: str, parameter_name: str) -> str:
    # Name with operation and parameter
    op_parameter_name = ''.join((operation_name, ' | ', parameter_name))

    # Name with node id || operation | parameter
    node_op_parameter_name = ''.join((str(node_id), ' || ', op_parameter_name))
    return node_op_parameter_name


def convert_parameters(parameters):
    """
    Function removes labels from dictionary with operations

    Args:
        parameters: labeled parameters

    Returns:
        new_parameters: dictionary without labels of node_id and operation_name
    """

    new_parameters = {}
    for operation_parameter, value in parameters.items():
        # Remove right part of the parameter name
        parameter_name = operation_parameter.split(' | ')[-1]

        if value is not None:
            new_parameters.update({parameter_name: value})

    return new_parameters
