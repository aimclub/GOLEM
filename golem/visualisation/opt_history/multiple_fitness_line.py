import os
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Union, Sequence

import numpy as np
from matplotlib import pyplot as plt

from golem.core.log import default_log
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_history.arg_constraint_wrapper import ArgConstraintWrapper
from golem.visualisation.opt_history.fitness_line import setup_fitness_plot, find_best_running_fitness
from golem.visualisation.opt_history.utils import show_or_save_figure


class MultipleFitnessLines(metaclass=ArgConstraintWrapper):
    """ Class to compare fitness changes during optimization process.
    :param histories_to_compare: dictionary with labels to display as keys and list of best finte as values. """

    def __init__(self,
                 historical_fitnesses: Dict[str, Sequence[Sequence[Union[float, Sequence[float]]]]],
                 metric_names,
                 visuals_params: Dict[str, Any] = None):
        self.historical_fitnesses = historical_fitnesses
        self.metric_names = metric_names
        self.visuals_params = visuals_params or {}
        self.log = default_log(self)

    @staticmethod
    def from_saved_histories(root_path: str):
        historical_fitnesses = dict.fromkeys(os.listdir(root_path))
        for exp_name in os.listdir(root_path):
            if historical_fitnesses[exp_name] is None:
                historical_fitnesses[exp_name] = []
            path_to_setup = os.path.join(root_path, exp_name)
            for dataset in os.listdir(path_to_setup):
                if dataset != 'tree_100':
                    continue
                path_to_dataset = os.path.join(path_to_setup, dataset)
                for launch_num in os.listdir(path_to_dataset):
                    if not launch_num.isdigit():
                        continue
                    path_to_launch = os.path.join(path_to_dataset, launch_num)
                    for file in os.listdir(path_to_launch):
                        if file.startswith('history'):
                            history = OptHistory.load(os.path.join(path_to_launch, file))
                            historical_fitnesses[exp_name].append(history.historical_fitness)
                            print(f"Loaded history for {launch_num} launch")
                print(f'Loaded {len(historical_fitnesses[exp_name])} trial histories for experiment: '
                      f'{exp_name} and dataset {dataset}')
            metric_names = history.objective.metric_names

        return MultipleFitnessLines(historical_fitnesses, metric_names)

    @staticmethod
    def from_histories(histories_to_compare: Dict[str, Sequence['OptHistory']]):
        for key, histories in histories_to_compare.items():
            histories_to_compare.update({key: [history.historical_fitness for history in histories]})
        metric_names = list(histories_to_compare.values())[0][0].objective.metric_names

        return MultipleFitnessLines(histories_to_compare, metric_names)

    def visualize(self,
                  save_path: Optional[Union[os.PathLike, str]] = None,
                  with_confidence: bool = True,
                  metric_id: int = 0,
                  dpi: Optional[int] = None):
        """ Visualizes the best fitness values during the evolution in the form of line.
        :param save_path: path to save the visualization. If set, then the image will be saved,
            and if not, it will be displayed.
        :param with_confidence: bool param specifying to use confidence interval or not.
        :param metric_id: numeric index of the metric to visualize (for multi-objective opt-n).
        :param dpi: DPI of the output figure.
        """
        save_path = save_path or self.get_predefined_value('save_path')
        dpi = dpi or self.get_predefined_value('dpi')

        fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
        xlabel = 'Generation'
        self.plot_multiple_fitness_lines(ax, metric_id, with_confidence)
        setup_fitness_plot(ax, xlabel, title=f'Fitness lines for {self.metric_names[metric_id]}')
        plt.legend()
        show_or_save_figure(fig, save_path, dpi)

    def plot_multiple_fitness_lines(self, ax: plt.axis, metric_id: int = 0, with_confidence: bool = True,
                                    path_to_save: str = None):
        for histories, label in zip(list(self.historical_fitnesses.values()), list(self.historical_fitnesses.keys())):
            plot_average_fitness_line_per_generations(ax, histories, label,
                                                      with_confidence=with_confidence,
                                                      metric_id=metric_id,
                                                      path_to_save=path_to_save)

    def get_predefined_value(self, param: str):
        return self.visuals_params.get(param)


def plot_average_fitness_line_per_generations(
        axis: plt.Axes,
        historical_fitnesses: Sequence[Sequence[Union[float, Sequence[float]]]],
        label: Optional[str] = None,
        metric_id: int = 0,
        with_confidence: bool = True,
        z_score: float = 1.96,
        path_to_save: str = None):
    """Plots average fitness line per number of histories
    with confidence interval for given z-score (default z=1.96 is for 95% confidence)."""

    trial_fitnesses: List[List[float]] = []
    for fitnesses in historical_fitnesses:
        best_fitnesses = find_best_running_fitness(fitnesses, metric_id)
        trial_fitnesses.append(best_fitnesses)

    # Get average fitness value with confidence values
    average_fitness_per_gen = []
    confidence_fitness_per_gen = []
    max_generations = max(len(i) for i in trial_fitnesses)
    for i in range(max_generations):
        all_fitness_gen = []
        for fitnesses in trial_fitnesses:
            if i < len(fitnesses):
                all_fitness_gen.append(fitnesses[i])
            else:
                all_fitness_gen.append(fitnesses[-1])
        average_fitness_per_gen.append(mean(all_fitness_gen))
        confidence = stdev(all_fitness_gen) / np.sqrt(len(all_fitness_gen)) \
            if len(all_fitness_gen) >= 2 else 0.
        confidence_fitness_per_gen.append(confidence)

    # Compute confidence interval
    xs = np.arange(len(average_fitness_per_gen))
    ys = np.array(average_fitness_per_gen)
    ci = z_score * np.array(confidence_fitness_per_gen)

    axis.plot(xs, average_fitness_per_gen, label=label)
    if with_confidence:
        axis.fill_between(xs, (ys - ci), (ys + ci), alpha=.2)
