import os
import tarfile

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal, ttest_ind

from experiments.experiment_analyzer import ExperimentAnalyzer
from golem.core.paths import project_root


if __name__ == '__main__':
    """ The result of analysis can be seen without running the script in
    '~/GOLEM/examples/experiment_analyzer/result_analysis.tar.gz'. """

    path_to_root = os.path.join(project_root(), 'examples', 'experiment_analyzer')

    # extract data if there is an archive
    if 'data.tar.gz' in os.listdir(path_to_root):
        tar = tarfile.open(os.path.join(path_to_root, 'data.tar.gz'), "r:gz")
        tar.extractall()
        tar.close()

    path_to_experiment_data = os.path.join(path_to_root, 'data')
    path_to_save = os.path.join(path_to_root, 'result_analysis')

    analyzer = ExperimentAnalyzer(path_to_root=path_to_experiment_data, folders_to_ignore=['result_analysis',
                                                                                           'Thumbs.db'])

    # to get convergence table with mean values
    path_to_save_convergence = os.path.join(path_to_save, 'convergence')

    convergence_mean = analyzer.analyze_convergence(history_folder='histories', is_raise=False,
                                                    path_to_save=path_to_save_convergence,
                                                    is_mean=True)

    # to get convergence boxplots
    convergence = analyzer.analyze_convergence(history_folder='histories', is_raise=False)
    path_to_save_convergence_boxplots = os.path.join(path_to_save_convergence, 'convergence_boxplots')

    metrics = list(convergence.keys())
    setups = list(convergence[metrics[0]].keys())
    datasets = list(convergence[metrics[0]][setups[0]].keys())
    for dataset in datasets:
        for metric_name in convergence.keys():
            to_compare = dict()
            for setup in convergence[metric_name].keys():
                to_compare[setup] = [i for i in convergence[metric_name][setup][dataset]]
            plt.boxplot(list(to_compare.values()), labels=list(to_compare.keys()))
            plt.title(f'Convergence on {dataset}')
            os.makedirs(path_to_save_convergence_boxplots, exist_ok=True)
            plt.savefig(os.path.join(path_to_save_convergence_boxplots, f'convergence_{dataset}.png'))
            plt.close()

    # to get metrics table with mean values
    path_to_save_metrics = os.path.join(path_to_save, 'metrics')
    metric_names = ['roc_auc', 'f1']
    metrics_dict_mean = analyzer.analyze_metrics(metric_names=metric_names, file_name='evaluation_results.csv',
                                                 is_raise=False, path_to_save=path_to_save_metrics,
                                                 is_mean=True)

    # to get metrics boxplots
    metrics_dict = analyzer.analyze_metrics(metric_names=metric_names, file_name='evaluation_results.csv',
                                            is_raise=False)
    path_to_save_metrics_boxplots = os.path.join(path_to_save_metrics, 'metrics_boxplot')

    for metric in metric_names:
        for dataset in metrics_dict[metric][list(metrics_dict[metric].keys())[0]].keys():
            to_compare = dict()
            for setup in metrics_dict[metric].keys():
                to_compare[setup] = [-1 * i for i in metrics_dict[metric][setup][dataset]]
            plt.boxplot(list(to_compare.values()), labels=list(to_compare.keys()))
            plt.title(f'{metric} on {dataset}')
            cur_path_to_save = os.path.join(path_to_save_metrics_boxplots, metric)
            os.makedirs(cur_path_to_save, exist_ok=True)
            plt.savefig(os.path.join(cur_path_to_save, f'{metric}_{dataset}.png'))
            plt.close()

    # to get stat test results table
    path_to_save_stat = os.path.join(path_to_save, 'statistic')
    stat_dict = analyzer.analyze_statistical_significance(data_to_analyze=metrics_dict['roc_auc'],
                                                          stat_tests=[mannwhitneyu, kruskal, ttest_ind],
                                                          path_to_save=path_to_save_stat)
