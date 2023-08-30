import os

from typing import Dict, List, Tuple

from golem.core.optimisers.opt_history_objects.opt_history import OptHistory


class ExperimentAnalyzer:
    """ Class to analyze results of experiment.
    To use this class folder hierarchy must be organized as following:

    setup (e.g. configuration of framework)
        \
        dataset
               \
               launch
                     \
                      all collected data (metrics.csv, history, saved_pipeline, etc)

    :param path_to_root: path to dir with experiment setups
    """

    def __init__(self, path_to_root: str):
        self.path_to_root = path_to_root

    def analyze_convergence(self, history_folder: str = 'history', is_raise: bool = False) \
            -> Dict[str, Dict[str, List[float]]]:
        """ Method to analyze convergence with the use of histories.

        :param history_folder: name of the history folder in experiment result folder (e.g. 'history', 'histories')
        :param is_raise: bool specifying if exception must be raised if there is no history folder
        """
        convergence = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():
            convergence = self._extend_result_dict(result_dict=convergence, setup=setup, dataset=dataset)

            if history_folder not in os.listdir(path_to_launch):
                if is_raise:
                    raise ValueError(f"There is no history folder with name {history_folder}")
                else:
                    continue

            path_to_history_folder = os.path.join(path_to_launch, history_folder)
            history_files = [file for file in os.listdir(path_to_history_folder) if file.endswith('.json')]

            # if there is no history
            if len(history_files) == 0:
                continue

            # load the first history in the folder
            history = OptHistory.load(os.path.join(path_to_history_folder, history_files[0]))
            convergence[setup][dataset].append(self._analyze_convergence(history=history))
        return convergence

    @staticmethod
    def _analyze_convergence(history: OptHistory) -> float:
        """ Method to get time in what individual with the best fitness was firstly obtained. """
        best_fitness = history.final_choices.data[0].fitness
        first_gen_with_best_fitness = history.generations_count
        for i, gen_fitnesses in enumerate(history.historical_fitness):
            if best_fitness in gen_fitnesses:
                first_gen_with_best_fitness = i
                break
        total_time_to_get_best_fitness = 0
        for i, gen in enumerate(history.individuals):
            if i == first_gen_with_best_fitness:
                break
            for ind in gen.data:
                total_time_to_get_best_fitness += ind.metadata['computation_time_in_seconds']
        return total_time_to_get_best_fitness

    def analyze_metrics(self):
        pass

    def analyze_structural_complexity(self):
        pass

    def _get_path_to_launch(self) -> Tuple[str, str, str]:
        """ Yields setup name, dataset name + paths to dirs with experiment results. """
        for setup in os.listdir(self.path_to_root):
            path_to_setup = os.path.join(self.path_to_root, setup)
            for dataset in os.listdir(path_to_setup):
                path_to_dataset = os.path.join(path_to_setup, dataset)
                for launch in os.listdir(path_to_dataset):
                    path_to_launch = os.path.join(path_to_dataset, launch)
                    yield setup, dataset, path_to_launch

    @staticmethod
    def _extend_result_dict(result_dict: dict, setup: str, dataset: str) -> Dict[str, Dict[str, list]]:
        """ Extends result dictionary with new setup and dataset name. """
        if setup not in result_dict.keys():
            result_dict[setup] = dict()
        if dataset not in result_dict[setup].keys():
            result_dict[setup][dataset] = []
        return result_dict
