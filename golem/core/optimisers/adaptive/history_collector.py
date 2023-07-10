import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence, Type, Optional, Any, Union, Generator

from golem.core.dag.graph import Graph
from golem.core.log import default_log
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimization_parameters import OptimizationParameters
from golem.core.optimisers.optimizer import GraphOptimizer, AlgorithmParameters, GraphGenerationParams
from golem.core.utilities.data_structures import update_dataclass


class HistoryCollector:
    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Union[Graph, Any]],
                 optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                 requirements: Optional[OptimizationParameters] = None,
                 graph_generation_params: Optional[GraphGenerationParams] = None,
                 graph_optimizer_params: Optional[AlgorithmParameters] = None,
                 save_path: Optional[Path] = None,
                 ):
        self.log = default_log(self)
        self.save_path = save_path or Path("results")
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.objective = objective
        self.initial_graphs = initial_graphs
        self.optimizer_cls = optimizer_cls
        self.requirements = requirements or OptimizationParameters()
        self.graph_generation_params = graph_generation_params or GraphGenerationParams()
        self.graph_optimizer_params = graph_optimizer_params or AlgorithmParameters()

    def collect_histories(self,
                          num_trials: int = 1,
                          trial_timeout: Optional[int] = None,
                          trial_iterations: Optional[int] = None,
                          experiment_id_prefix: str = 'collect'
                          ):
        # Construct experiment id
        id_parts = [datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                    experiment_id_prefix,
                    self.optimizer_cls.__name__.lower(),
                    f'min{trial_timeout}' if trial_timeout else None,
                    f'iters{trial_iterations}' if trial_iterations else None,
                    ]
        experiment_id = '_'.join(filter(None, id_parts))

        # Construct common parameters for all runs
        base_run_params = dict(
            num_of_generations=trial_iterations,
            timeout=timedelta(minutes=trial_timeout) if trial_timeout else None,
            early_stopping_timeout=trial_timeout // 2 if trial_timeout else None,
            early_stopping_iterations=trial_iterations // 2 if trial_iterations else None,
            keep_history=True,  # use OptHistory for logging
            history_dir=None,  # don't save intermediate results; we need final whole history
            n_jobs=-1,
        )
        # and update user graph requirements with them
        graph_requirements = update_dataclass(self.requirements, update_dc=base_run_params)

        for i in range(num_trials):
            start_time = datetime.now()
            self.log.info(f'\nTrial #{i + 1} of {experiment_id} started at {start_time}')

            # Run optimizer setup
            optimizer = self.optimizer_cls(objective=self.objective,
                                           initial_graphs=self.initial_graphs,
                                           requirements=graph_requirements,
                                           graph_generation_params=self.graph_generation_params,
                                           graph_optimizer_params=self.graph_optimizer_params)

            history_filename = f'history_trial{i}.json'
            optimizer.history.save(self.save_path / history_filename)

            duration = datetime.now() - start_time
            self.log.info(f'Trial #{i + 1}/{num_trials} finished, spent time: {duration}')
        self.log.info(f'Saved {num_trials} trial histories to dir: {self.save_path.absolute()}')


class HistoryReader:
    def __init__(self, save_path: Optional[Path] = None):
        self.log = default_log(self)
        self.save_path = save_path or Path("results")
        self.save_path.mkdir(parents=True, exist_ok=True)

    def load_histories(self) -> Generator[OptHistory]:
        """Iteratively loads saved histories one-by-ony."""
        num_histories = 0
        total_individuals = 0
        for history_path in HistoryReader.traverse_histories(self.save_path):
            history = OptHistory.load(history_path)
            num_histories += 1
            total_individuals += sum(map(len, history.generations))
            yield history

        if num_histories == 0 or total_individuals == 0:
            raise ValueError(f'Could not load any individuals.'
                             f'Possibly, path {self.save_path} does not exist or is empty.')
        else:
            self.log.info(f'Loaded {num_histories} histories '
                          f'with {total_individuals} individuals in total.')

    @staticmethod
    def traverse_histories(path) -> Generator[Path]:
        if path.exists():
            # recursive traversal of the save directory
            for root, dirs, files in os.walk(path):
                for history_filename in files:
                    if history_filename.startswith('history'):
                        full_path = Path(root) / history_filename
                        yield full_path
