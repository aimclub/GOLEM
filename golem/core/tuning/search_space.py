from typing import Dict, Tuple, Callable, List


class SearchSpace:
    """
    Args:
        search_space: dictionary with parameters and their search_space
            {'operation_name': {'param_name': (hyperopt distribution function, [sampling scope]), ...}, ...},
            e.g. ``{'operation_name': {'param1': (hp.uniformint, [2, 21]), ...}, ..}
    """
    def __init__(self, search_space: Dict[str, Dict[str, Tuple[Callable, List]]]):
        self.parameters_per_operation = search_space

    def get_operation_parameter_range(self, operation_name: str, parameter_name: str = None, label: str = 'default'):
        """
        Method return hyperopt object with search_space from search_space dictionary
        If parameter name is not defined - return all available operations

        Args:
            operation_name: name of the operation
            parameter_name: name of hyperparameter of particular operation
            label: label to assign in hyperopt pyll

        Returns:
            dictionary with appropriate range
        """

        # Get available parameters for current operation
        operation_parameters = self.parameters_per_operation.get(operation_name)

        if operation_parameters is not None:
            # If there are not parameter_name - return list with all parameters
            if parameter_name is None:
                return list(operation_parameters)
            else:
                hyperopt_tuple = operation_parameters.get(parameter_name)
                return hyperopt_tuple[0](label, *hyperopt_tuple[1])
        else:
            return None

    def get_node_params(self, node_id, operation_name):
        """
        Method for forming dictionary with hyperparameters for considering
        operation as a part of the whole graph

        :param node_id: number of node in graph.nodes list
        :param operation_name: name of operation in the node

        :return params_dict: dictionary-like structure with labeled hyperparameters
        and their range per operation
        """

        # Get available parameters for operation
        params_list = self.get_operation_parameter_range(operation_name)

        if params_list is None:
            params_dict = None
        else:
            params_dict = {}
            for parameter_name in params_list:
                node_op_parameter_name = get_node_operation_parameter_label(node_id, operation_name, parameter_name)

                # For operation get range where search can be done
                space = self.get_operation_parameter_range(operation_name=operation_name,
                                                           parameter_name=parameter_name,
                                                           label=node_op_parameter_name)

                params_dict.update({node_op_parameter_name: space})

        return params_dict


def get_node_operation_parameter_label(node_id: int, operation_name: str, parameter_name: str) -> str:
    # Name with operation and parameter
    op_parameter_name = ''.join((operation_name, ' | ', parameter_name))

    # Name with node id || operation | parameter
    node_op_parameter_name = ''.join((str(node_id), ' || ', op_parameter_name))
    return node_op_parameter_name


def convert_params(params):
    """
    Function removes labels from dictionary with operations

    :param params: labeled parameters
    :return new_params: dictionary without labels of node_id and operation_name
    """

    new_params = {}
    for operation_parameter, value in params.items():
        # Remove right part of the parameter name
        parameter_name = operation_parameter.split(' | ')[-1]

        if value is not None:
            new_params.update({parameter_name: value})

    return new_params
