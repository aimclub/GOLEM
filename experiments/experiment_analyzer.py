import os
from statistics import mean

from typing import Dict, List, Tuple, Any, Callable, Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from examples.adaptive_optimizer.utils import plot_action_values
from golem.core.log import default_log
from golem.core.optimisers.adaptive.operator_agent import OperatorAgent, ObsType
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines


class ExperimentAnalyzer:
    """ Class to analyze results of experiment.
    The example of usage can be found here: ~/GOLEM/examples/experiment_analyzer/
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

    def __init__(self, path_to_root: str, folders_to_ignore: Optional[List[str]] = None):
        if folders_to_ignore is None:
            folders_to_ignore = []
        self.path_to_root = path_to_root
        self._folders_to_ignore = folders_to_ignore
        self._log = default_log('ExperimentAnalyzer')

    def analyze_convergence(self, history_folder: Optional[str] = 'history', is_mean: bool = False,
                            path_to_save: Optional[str] = None, is_raise: bool = False,
                            metrics_names: Optional[List[str]] = None) \
            -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """ Method to analyze convergence with the use of histories.

        :param history_folder: name of the history folder in experiment result folder (e.g. 'history', 'histories').
        If history is not in separate folder than it must be specified as None.
        :param is_mean: bool flag to specify just storing all the results or calculating mean values
        :param path_to_save: path to dir where to save results
        :param is_raise: bool specifying if exception must be raised if there is no history folder
        :param metrics_names: names of metrics to consider. Useful in case of multimodal optimization.
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)

        if metrics_names is None:
            metrics_names = ['metric']

        convergence = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():

            for metric in metrics_names:
                convergence.setdefault(metric, dict())
                convergence[metric] = \
                    self._extend_result_dict(result_dict=convergence[metric], setup=setup, dataset=dataset)

            path_to_history_folder = os.path.join(path_to_launch, history_folder) if history_folder else path_to_launch
            if not self._check_if_file_or_folder_present(path=path_to_history_folder, is_raise=is_raise):
                continue

            history_files = [file for file in os.listdir(path_to_history_folder) if file.endswith('.json')]

            # if there is no history
            if len(history_files) == 0:
                continue

            # load the first history in the folder
            history = OptHistory.load(os.path.join(path_to_history_folder, history_files[0]))

            convergence_per_metric = self._analyze_convergence(history=history,
                                                               metrics_num=len(metrics_names))
            for i, metric in enumerate(metrics_names):
                convergence[metric][setup][dataset].append(convergence_per_metric[i])

        if is_mean:
            for metric in convergence.keys():
                for setup in convergence[metric].keys():
                    for dataset in convergence[metric][setup].keys():
                        convergence[metric][setup][dataset] = mean(convergence[metric][setup][dataset])

        # save results per metric
        if path_to_save:
            for metric in metrics_names:
                df = pd.DataFrame(convergence[metric])
                path_to_save = os.path.join(path_to_save, f'convergence_results_{metric}.csv')
                df.to_csv(path_to_save)
                self._log.info(f"Convergence table was saved to {path_to_save}")
        return convergence

    @staticmethod
    def _analyze_convergence(history: OptHistory, metrics_num: int) -> List[float]:
        """ Method to get time in what individual with the best fitness was firstly obtained. """
        # find best final metric for each objective
        best_fitness_per_objective = [np.inf] * metrics_num
        for i in range(metrics_num):
            for ind in history.final_choices.data:
                if ind.fitness.values[i] < best_fitness_per_objective[i]:
                    best_fitness_per_objective[i] = ind.fitness.values[i]

        total_time_to_get_best_fitness_per_objective = []
        for j, best_fitness in enumerate(best_fitness_per_objective):
            first_gen_with_best_fitness = history.generations_count
            # since if objective is not multi-modal than historical_fitness[j] is 1d list
            historical_fitness = history.historical_fitness[j]
            if not history.objective.is_multi_objective:
                historical_fitness = [history.historical_fitness[j]]

            for i, gen_fitnesses in enumerate(historical_fitness):
                if best_fitness in gen_fitnesses:
                    first_gen_with_best_fitness = i
                    break
            total_time_to_get_best_fitness = 0
            for i, gen in enumerate(history.individuals):
                if i == first_gen_with_best_fitness:
                    break
                for ind in gen.data:
                    total_time_to_get_best_fitness += ind.metadata['computation_time_in_seconds']
            total_time_to_get_best_fitness_per_objective.append(total_time_to_get_best_fitness)
        return total_time_to_get_best_fitness_per_objective

    def analyze_metrics(self, metric_names: List[str], file_name: str, is_mean: bool = False,
                        path_to_save: Optional[str] = None, is_raise: bool = False, keep_n_best: Optional[int] = 1) \
            -> Dict[str, Dict[str, Dict[str, Union[List[float], float]]]]:
        """ Method to analyze specified metrics.
        :param metric_names: names of metrics to analyze. e.g. ['f1', 'inference_time']
        :param file_name: name of the file with metrics (e.g. 'metrics.csv').
        The file with metrics must have metric names in columns.
        :param is_mean: bool flag to specify just storing all the results or calculating mean values
        :param path_to_save: path to dir where to save results
        :param is_raise: bool specifying if exception must be raised if there is no history folder
        :param keep_n_best: number of result to consider when calculating metric.
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)

        dict_with_metrics = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():

            df_metrics = self._get_metrics_df_from_path(path=path_to_launch, file_name=file_name, is_raise=is_raise)
            if df_metrics.empty:
                continue

            for metric in metric_names:
                dict_with_metrics.setdefault(metric, dict())
                dict_with_metrics[metric] = self._extend_result_dict(result_dict=dict_with_metrics[metric],
                                                                     setup=setup, dataset=dataset)
                if metric not in df_metrics.columns:
                    self._log.warning(f"There is no column in {file_name} with {metric}")

                # if there were multiple metrics in experiment or not
                all_metric_values = list(df_metrics[metric])
                all_metric_values.sort()
                dict_with_metrics[metric][setup][dataset].append(mean(all_metric_values[:keep_n_best]))

        if is_mean:
            for metric in metric_names:
                for setup in dict_with_metrics[metric].keys():
                    for dataset in dict_with_metrics[metric][setup].keys():
                        dict_with_metrics[metric][setup][dataset] = mean(dict_with_metrics[metric][setup][dataset])

        # save results per metric
        if path_to_save:
            for metric in dict_with_metrics.keys():
                df = pd.DataFrame(dict_with_metrics[metric])
                cur_path_to_save = os.path.join(path_to_save, f'{metric}_results.csv')
                df.to_csv(cur_path_to_save)
                self._log.info(f"Metric table was saved to {cur_path_to_save}")
        return dict_with_metrics

    def plot_convergence(self, path_to_save: str, with_confidence: bool = True,
                         history_folder: str = 'history', is_raise: bool = False,
                         metrics_names: Optional[List[str]] = None):
        """ Method to analyze convergence with the use of histories.

        :param path_to_save: path to dir where to save results
        :param with_confidence: bool param specifying to use confidence interval or not.
        :param history_folder: name of the history folder in experiment result folder (e.g. 'history', 'histories')
        :param is_raise: bool specifying if exception must be raised if there is no history folder.
        :param metrics_names: names of metrics to consider. Useful in case of multimodal optimization.
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)

        if not metrics_names:
            metrics_names = ['metric']

        histories = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():
            histories = self._extend_result_dict(result_dict=histories, setup=setup, dataset=dataset)

            path_to_history_folder = os.path.join(path_to_launch, history_folder)
            if not self._check_if_file_or_folder_present(path=path_to_history_folder, is_raise=is_raise):
                continue

            history_files = [file for file in os.listdir(path_to_history_folder) if file.endswith('.json')]

            # if there is no history
            if len(history_files) == 0:
                continue

            # load the first history in the folder
            history = OptHistory.load(os.path.join(path_to_history_folder, history_files[0]))
            self._log.info(f"Loaded {history_files[0]} history for {setup} and {dataset} dataset")
            histories[setup][dataset].append(history.historical_fitness)

        histories_to_compare = dict()
        # plot convergence pics
        for dataset in histories[list(histories.keys())[0]]:
            for setup in histories.keys():
                histories_to_compare[setup] = histories[setup][dataset] if dataset in histories[setup].keys() else []
            multiple_fitness_plot = MultipleFitnessLines(historical_fitnesses=histories_to_compare,
                                                         metric_names=metrics_names)
            for i, metric in enumerate(metrics_names):
                file_name = f'{dataset}_convergence_with_confidence_for_{metric}' if with_confidence \
                    else f'{dataset}_convergence_without_confidence_for_{metric}'
                cur_path_to_save = os.path.join(path_to_save, file_name)
                multiple_fitness_plot.visualize(save_path=cur_path_to_save,
                                                with_confidence=with_confidence, metric_id=i)
                self._log.info(f"Convergence plot for {dataset} dataset for {metric} was saved to {cur_path_to_save}")

    def analyze_statistical_significance(self, data_to_analyze: Dict[str, Dict[str, List[float]]],
                                         stat_tests: List[Callable], path_to_save: Optional[str] = None,
                                         test_format: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """ Method to perform statistical analysis of data. Metric data obtained with 'analyze_metrics' and
        convergence data obtained with 'analyze_convergence' can be simply analyzed, for example.
        :param data_to_analyze: data to analyze.
        NB! data must have the specified format Dict[str, Dict[str, float]]:
        first key -- framework/setup name, second -- dataset name and then list of metric values
        :param stat_tests: list of functions of statistical tests to perform. E.g. scipy.stats.kruskal
        :param path_to_save: path to dir where to save results
        :param test_format: argument to specify what every test function must return. default: ['statistic', 'pvalue']
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)

        if not test_format:
            test_format = ['statistic', 'pvalue']

        stat_dict = dict.fromkeys(test_format, None)
        datasets = list(data_to_analyze[list(data_to_analyze.keys())[0]].keys())
        for dataset in datasets:
            values_to_compare = []
            for setup in data_to_analyze.keys():
                values_to_compare.append(data_to_analyze[setup][dataset])
            for test in stat_tests:
                try:
                    cur_test_result = test(*values_to_compare)
                except Exception as e:
                    self._log.critical(f"Statistical test ({test}) failed with exception: {e}")
                    cur_test_result = [None] * len(test_format)
                for i, arg in enumerate(test_format):
                    if not stat_dict[arg]:
                        stat_dict[arg] = dict.fromkeys([t.__name__ for t in stat_tests], None)
                    if not stat_dict[arg][test.__name__]:
                        stat_dict[arg][test.__name__] = dict.fromkeys(datasets, None)
                    stat_dict[arg][test.__name__][dataset] = cur_test_result[i]

        # save results for all tests
        if path_to_save:
            for arg in test_format:
                df = pd.DataFrame(stat_dict[arg])
                cur_path_to_save = os.path.join(path_to_save, f'stat_{arg}_results.csv')
                df.to_csv(cur_path_to_save)
                self._log.info(f"Stat test table for {arg} was saved to {cur_path_to_save}")
        return stat_dict

    def analyze_structural_complexity(self, path_to_save: str, dir_name: str, class_to_load: Any = None,
                                      load_func: Optional[Callable] = None, is_raise: bool = False,
                                      file_name: Optional[str] = None, metrics_to_display: List[str] = None):
        """ Method to save pictures of final graphs in directories to compare it visually.
        :param path_to_save: root path to pictures per setup.
        :param dir_name: name of directory in which final graph is saved.
        :param class_to_load: class of objects to load.
        :param load_func: function that load object. Can be used in case method 'load' is not defined for the object.
        :param is_raise: bool specifying if exception must be raised if there is no history folder.
        :param file_name: name of the file with metrics (e.g. 'metrics.csv')
        :param metrics_to_display: list of metrics to display in the title of the picture with result.
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)

        for setup, dataset, path_to_launch in self._get_path_to_launch():
            if not self._check_if_file_or_folder_present(path=os.path.join(path_to_launch, dir_name), is_raise=is_raise):
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

            result = class_to_load.load(path_to_json) if class_to_load is not None \
                else load_func(path_to_json) if load_func else None

            # load metrics for title if specified
            df_metrics = self._get_metrics_df_from_path(path=path_to_launch, file_name=file_name, is_raise=is_raise)
            title = ''
            for metric in metrics_to_display:
                if metric not in df_metrics.columns:
                    self._log.warning(f"There is no column in {file_name} with {metric}")
                else:
                    title += f'{metric}={df_metrics[metric][0]} '
            title = 'Best metrics for launch: ' + title
            cur_path_to_save = os.path.join(path_to_save, setup)

            if not os.path.exists(cur_path_to_save):
                os.makedirs(cur_path_to_save)
            saved_results = [int(cur_name.split("_")[0]) for cur_name in os.listdir(cur_path_to_save)
                             if cur_name not in self._folders_to_ignore]
            max_saved_num = max(saved_results) if saved_results else 0
            cur_path_to_save = os.path.join(cur_path_to_save, f'{max_saved_num+1}_result.png')
            result.show(cur_path_to_save, title=title)
            self._log.info(f"Resulting graph was saved to {cur_path_to_save}")

    def plot_agent_mutation_probs_and_expectations(self, path_to_save: str, agent_class: type(OperatorAgent),
                                                   mutation_names: List[str], gen_to_save_agent: int = 5,
                                                   dir_name: str = 'agent', is_raise: bool = False,
                                                   obs: Optional[ObsType] = None):
        """ Analyzes agent mutation probabilities and actions expectations through the evolution process.
        Useful for simple MAB and contextual MABs.
        :param path_to_save:
        :param agent_class: class of agent to load
        :param mutation_names: mutations that were applied by agent
        :param gen_to_save_agent: the agent is saved every generation which % gen_to_save_agent == 0.
        Originally, the agent is saved every 5th generation to not overload memory.
        :param dir_name: name of directory in which agent are saved.
        :param is_raise: bool specifying if exception must be raised if there is no history folder.
        :param obs: observation to throw to agent. E.g. ContextualMultiArmedBanditAgent needs an
        observation to predict action probabilities. If the specified agent does not need observation,
        can be left as None.
        """
        if path_to_save:
            os.makedirs(path_to_save, exist_ok=True)

        probs_dict = dict()
        for setup, dataset, path_to_launch in self._get_path_to_launch():
            if not self._check_if_file_or_folder_present(path=os.path.join(path_to_launch, dir_name),
                                                         is_raise=is_raise):
                continue

            probs_dict = self._extend_result_dict(result_dict=probs_dict, setup=setup, dataset=dataset)

            path_to_agents = os.path.join(path_to_launch, dir_name)
            cur_probabilities_list = []
            for agent_num in os.listdir(path_to_agents):
                path_cur_agent = os.path.join(path_to_agents, agent_num)
                agent = agent_class.load(path_cur_agent)
                probs = agent.get_action_values(obs)
                cur_probabilities_list.append(probs)
            probs_dict[setup][dataset].append(cur_probabilities_list)
            self._log.info(f"Agents for {setup} {dataset} were analyzed")

        # Average out probabilities per generation
        num_mutations = len(mutation_names)
        average_probabilities = {}
        for setup in probs_dict.keys():
            for dataset in probs_dict[setup].keys():
                average_probs = []
                all_probabilities = probs_dict[setup][dataset]
                max_gen = max([len(agent) for agent in all_probabilities])
                for gen in range(max_gen):
                    all_probs_per_gen = []
                    for i in range(num_mutations):
                        all_probs_per_gen.append([])
                    for agent in all_probabilities:
                        for mutation_num in range(num_mutations):
                            # extend mutations as if all launches are of the same length
                            if gen >= len(agent):
                                all_probs_per_gen[mutation_num].append(agent[-1][mutation_num])
                            else:
                                all_probs_per_gen[mutation_num].append(agent[gen][mutation_num])
                    average_probs_per_gen = []
                    for probs_list in all_probs_per_gen:
                        average_probs_per_gen.append(mean(probs_list))

                    for i in range(gen_to_save_agent):
                        average_probs.append(average_probs_per_gen)
                average_probabilities = self._extend_result_dict(result_dict=average_probabilities,
                                                                 setup=setup, dataset=dataset)
                average_probabilities[setup][dataset] = average_probs

                plot_action_values(stats=average_probabilities[setup][dataset], action_tags=mutation_names,
                                   titles=['Average action Expectation Values', 'Average action Probabilities'])
                cur_path_to_save = os.path.join(path_to_save, f'{setup}_{dataset}_probs.png')
                plt.savefig(cur_path_to_save)
                self._log.info(f"Plot of action expectations and mutation probabilities was saved: {cur_path_to_save}")

    def _get_path_to_launch(self) -> Tuple[str, str, str]:
        """ Yields setup name, dataset name and paths to dirs with experiment results.
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
        result_dict.setdefault(setup, dict())
        result_dict[setup].setdefault(dataset, list())
        return result_dict

    def _get_metrics_df_from_path(self, path: str, file_name: str, is_raise: bool) -> pd.DataFrame:
        path_to_df = os.path.join(path, file_name)
        if not self._check_if_file_or_folder_present(path_to_df, is_raise):
            return pd.DataFrame()

        df_metrics = pd.read_csv(path_to_df)
        return df_metrics

    @staticmethod
    def _check_if_file_or_folder_present(path: str, is_raise: bool) -> bool:
        if not (os.path.isdir(path) or os.path.isfile(path)):
            if is_raise:
                raise ValueError(f"There is no folder/file with name {os.path.split(path)[-1]}")
            else:
                return False
        return True
