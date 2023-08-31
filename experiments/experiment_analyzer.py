import os
from statistics import mean

from typing import Dict, List, Tuple, Any

import pandas as pd

from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_history.fitness_line import MultipleFitnessLines


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

    :param path_to_root: path to dir with experiment setups.
    :param folders_to_ignore: folders without experiment data to ignore.
    """

    def __init__(self, path_to_root: str, folders_to_ignore: List[str] = []):
        self.path_to_root = path_to_root
        self._folders_to_ignore = folders_to_ignore
        self._log = default_log('ExperimentAnalyzer')

    def analyze_convergence(self, history_folder: str = 'history', is_mean: bool = False,
                            path_to_save: str = None, is_raise: bool = False) \
            -> Dict[str, Dict[str, List[float]]]:
        """ Method to analyze convergence with the use of histories.

        :param history_folder: name of the history folder in experiment result folder (e.g. 'history', 'histories')
        :param is_mean: bool flag to specify just storing all the results or calculating mean values
        :param path_to_save: path to save results.
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

        if is_mean:
            for setup in convergence.keys():
                for dataset in convergence[setup].keys():
                    convergence[setup][dataset] = mean(convergence[setup][dataset])

        # save results per metric
        if path_to_save:
            df = pd.DataFrame(convergence)
            df.to_csv(os.path.join(path_to_save, f'convergence_results.csv'))
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

    def analyze_metrics(self, metric_names: List[str], file_name: str, is_mean: bool = False,
                        path_to_save: str = None, is_raise: bool = False):
        """ Method to analyze specified metrics.
        :param metric_names: names of metrics to analyze. e.g. ['f1', 'inference_time']
        :param file_name: name of the file with metrics (e.g. 'metrics.csv')
        :param is_mean: bool flag to specify just storing all the results or calculating mean values
        :param path_to_save: path to save results
        :param is_raise: bool specifying if exception must be raised if there is no history folder
        """
        dict_with_metrics = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():

            if file_name not in os.listdir(path_to_launch):
                if is_raise:
                    raise ValueError(f"There is no metric file with name {file_name}")
                else:
                    continue

            df_metrics = pd.read_csv(os.path.join(path_to_launch, file_name))

            for metric in metric_names:
                if metric not in dict_with_metrics.keys():
                    dict_with_metrics[metric] = dict()
                dict_with_metrics[metric] = self._extend_result_dict(result_dict=dict_with_metrics[metric],
                                                                     setup=setup, dataset=dataset)
                if metric not in df_metrics.columns:
                    self._log.warning(f"There is no column in {file_name} with {metric}")
                dict_with_metrics[metric][setup][dataset].append(df_metrics[metric][0])

        if is_mean:
            for metric in metric_names:
                for setup in dict_with_metrics[metric].keys():
                    for dataset in dict_with_metrics[metric][setup].keys():
                        dict_with_metrics[metric][setup][dataset] = mean(dict_with_metrics[metric][setup][dataset])

        # save results per metric
        if path_to_save:
            for metric in dict_with_metrics.keys():
                df = pd.DataFrame(dict_with_metrics[metric])
                df.to_csv(os.path.join(path_to_save, f'{metric}_results.csv'))
        return dict_with_metrics

    def plot_convergence(self, path_to_save: str,
                         history_folder: str = 'history', is_raise: bool = False):
        """ Method to analyze convergence with the use of histories.

        :param path_to_save: path to save the results.
        :param history_folder: name of the history folder in experiment result folder (e.g. 'history', 'histories')
        :param is_raise: bool specifying if exception must be raised if there is no history folder
        """
        histories = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():
            histories = self._extend_result_dict(result_dict=histories, setup=setup, dataset=dataset)

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
            histories[setup][dataset].append(history)

        histories_to_compare = dict()
        # plot convergence pics
        for dataset in histories[list(histories.keys())[0]]:
            for setup in histories.keys():
                histories_to_compare[setup] = histories[setup][dataset]
            multiple_fitness_plot = MultipleFitnessLines(histories_to_compare=histories_to_compare)
            cur_path_to_save = os.path.join(path_to_save, f'{dataset}_convergence_without_confidence')
            multiple_fitness_plot.visualize(save_path=cur_path_to_save)

    def analyze_structural_complexity(self, path_to_save: str, dir_name: str, class_to_load: Any,
                                      is_raise: bool = False):
        """ Method to save pictures of final graphs in directories to compare it visually.
        :param path_to_save: root path to pictures per setup.
        :param dir_name: name of directory in which final graph is saved.
        :param class_to_load: class of objects to load
        :param is_raise: bool specifying if exception must be raised if there is no history folder
        """
        for setup, dataset, path_to_launch in self._get_path_to_launch():
            if dir_name not in os.listdir(path_to_launch):
                if is_raise:
                    raise ValueError(f"There is no folder with name {dir_name}")
                else:
                    continue

            path_to_json = None
            for address, dirs, files in os.walk(path_to_launch):
                for name in files:
                    if '.json' in name:
                        path_to_json = os.path.join(address, name)
                        break

            # final result was not saved in this launch
            if not path_to_json:
                continue
            final = class_to_load.load(path_to_json)
            path_to_save = os.path.join(path_to_save, setup)

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            saved_results = [file_name.split("_")[0] for file_name in os.listdir(path_to_save)]
            max_saved_num = max(saved_results) if saved_results else 0
            final.show(os.path.join(path_to_save, f'{max_saved_num}_result.png'))

    def _get_path_to_launch(self) -> Tuple[str, str, str]:
        """ Yields setup name, dataset name + paths to dirs with experiment results.
        If experiment saving configuration/files structure somehow differs from the structure implied in this class
        this method can be used externally to get paths to launches.
        """
        for setup in os.listdir(self.path_to_root):
            if setup in self._folders_to_ignore:
                continue
            path_to_setup = os.path.join(self.path_to_root, setup)
            for dataset in os.listdir(path_to_setup):
                if dataset in self._folders_to_ignore:
                    continue
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
