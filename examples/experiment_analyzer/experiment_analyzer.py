import os

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, ttest_ind

from experiments.experiment_analyzer import ExperimentAnalyzer
from golem.core.paths import project_root


def create_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':

    path_to_root = os.path.join(project_root(), 'examples', 'experiment_analyzer')
    path_to_experiment_data = os.path.join(path_to_root, 'data')
    path_to_save = os.path.join(path_to_root, 'result_analysis')

    analyzer = ExperimentAnalyzer(path_to_root=path_to_experiment_data, folders_to_ignore=['result_analysis',
                                                                                           'Thumbs.db'])

    # to get convergence table with mean values
    path_to_save_convergence = os.path.join(path_to_save, 'convergence')
    create_if_not_exists(path_to_save_convergence)

    convergence_mean = analyzer.analyze_convergence(history_folder='histories', is_raise=False,
                                                    path_to_save=path_to_save_convergence,
                                                    is_mean=True)

    # to get convergence boxplots
    convergence = analyzer.analyze_convergence(history_folder='histories', is_raise=False)
    path_to_save_convergence_boxplots = os.path.join(path_to_save_convergence, 'convergence_boxplots')
    create_if_not_exists(path_to_save_convergence_boxplots)

    for dataset in convergence[list(convergence.keys())[0]].keys():
        to_compare = dict()
        for setup in convergence.keys():
            to_compare[setup] = [i for i in convergence[setup][dataset]]
        plt.boxplot(list(to_compare.values()), labels=list(to_compare.keys()))
        plt.title(f'Convergence on {dataset}')
        plt.savefig(os.path.join(path_to_save_convergence_boxplots, f'convergence_{dataset}.png'))
        plt.close()

    # to get metrics table with mean values
    path_to_save_metrics = os.path.join(path_to_save, 'metrics')
    create_if_not_exists(path_to_save_metrics)
    metric_names = ['roc_auc', 'f1']
    metrics_dict_mean = analyzer.analyze_metrics(metric_names=metric_names, file_name='evaluation_results.csv',
                                                 is_raise=False, path_to_save=path_to_save_metrics,
                                                 is_mean=True)

    # to get metrics boxplots
    metrics_dict = analyzer.analyze_metrics(metric_names=metric_names, file_name='evaluation_results.csv',
                                            is_raise=False)
    path_to_save_metrics_boxplots = os.path.join(path_to_save_metrics, 'metrics_boxplot')
    create_if_not_exists(path_to_save_metrics_boxplots)

    for metric in metric_names:
        for dataset in metrics_dict[metric][list(metrics_dict[metric].keys())[0]].keys():
            to_compare = dict()
            for setup in metrics_dict[metric].keys():
                to_compare[setup] = [-1 * i for i in metrics_dict[metric][setup][dataset]]
            plt.boxplot(list(to_compare.values()), labels=list(to_compare.keys()))
            plt.title(f'{metric} on {dataset}')
            cur_path_to_save = os.path.join(path_to_save_metrics_boxplots, metric)
            if not os.path.exists(cur_path_to_save):
                os.makedirs(cur_path_to_save)
            plt.savefig(os.path.join(cur_path_to_save, f'{metric}_{dataset}.png'))
            plt.close()

    # to get stat test results table
    path_to_save_stat = os.path.join(path_to_save, 'statistic')
    create_if_not_exists(path_to_save_stat)
    stat_dict = analyzer.analyze_statistical_significance(data_to_analyze=metrics_dict['roc_auc'],
                                                          stat_tests=[mannwhitneyu, kruskal, ttest_ind],
                                                          path_to_save=path_to_save_stat)