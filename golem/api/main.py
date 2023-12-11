import logging
from typing import Optional

from golem.api.api_utils.api_params import ApiParams
from golem.core.constants import DEFAULT_API_TIMEOUT_MINUTES
from golem.core.log import Log, default_log
from golem.utilities.utilities import set_random_seed


class GOLEM:
    """
    Main class for GOLEM API.

    Args:
        timeout: timeout for optimization.
        seed: value for a fixed random seed.
        logging_level: logging levels are the same as in `logging <https://docs.python.org/3/library/logging.html>`_.

            .. details:: Possible options:

                - ``50`` -> critical
                - ``40`` -> error
                - ``30`` -> warning
                - ``20`` -> info
                - ``10`` -> debug
                - ``0`` -> nonset
        n_jobs: num of ``n_jobs`` for parallelization (set to ``-1`` to use all cpu's). Defaults to ``-1``.
        graph_requirements_class: class to specify custom graph requirements.
        Must be inherited from GraphRequirements class.
    """
    def __init__(self,
                 timeout: Optional[float] = DEFAULT_API_TIMEOUT_MINUTES,
                 seed: Optional[int] = None,
                 logging_level: int = logging.ERROR,
                 n_jobs: int = -1,
                 **all_parameters):
        set_random_seed(seed)
        self.log = self._init_logger(logging_level)

        self.api_params = ApiParams(input_params=all_parameters,
                                    n_jobs=n_jobs,
                                    timeout=timeout)
        self.gp_algorithm_parameters = self.api_params.get_gp_algorithm_parameters()
        self.graph_generation_parameters = self.api_params.get_graph_generation_parameters()
        self.graph_requirements = self.api_params.get_graph_requirements()

    def optimise(self):
        common_params = self.api_params.get_actual_common_params()
        optimizer_cls = common_params['optimizer']
        objective = common_params['objective']
        initial_graphs = common_params['initial_graphs']
        graph_requirements = self.api_params.get_graph_requirements()
        graph_generation_parameters = self.api_params.get_graph_generation_parameters()
        gp_algorithm_parameters = self.api_params.get_gp_algorithm_parameters()

        optimiser = optimizer_cls(objective,
                                  initial_graphs,
                                  graph_requirements,
                                  graph_generation_parameters,
                                  gp_algorithm_parameters)

        found_graphs = optimiser.optimise(objective)
        return found_graphs

    @staticmethod
    def _init_logger(logging_level: int):
        # reset logging level for Singleton
        Log().reset_logging_level(logging_level)
        return default_log(prefix='GOLEM logger')
