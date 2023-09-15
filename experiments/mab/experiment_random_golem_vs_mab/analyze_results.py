import os

from scipy.stats import mannwhitneyu

from experiments.experiment_analyzer import ExperimentAnalyzer
from golem.core.paths import project_root

if __name__ == '__main__':
    path_to_root = os.path.join(project_root(), 'experiments', 'mab', 'experiment_random_golem_vs_mab', 'results')

    path_to_save = os.path.join(path_to_root, 'result_analysis')

    analyzer = ExperimentAnalyzer(path_to_root=path_to_root, folders_to_ignore=['result_analysis',
                                                                                'Thumbs.db'])
    # analyzer.analyze_metrics(metric_names=['sp_adj', 'sp_lapl', 'graph_size', 'degree'],
    #                          file_name='metrics.csv', is_mean=True,
    #                          path_to_save=path_to_save)

    convergence = analyzer.analyze_convergence(history_folder=None, is_mean=True, path_to_save=path_to_save)

    a = {}
    a['bandit'] = convergence['bandit']
    a['random'] = convergence['random']

    analyzer.analyze_statistical_significance(stat_tests=[mannwhitneyu],
                                              data_to_analyze=a, path_to_save=path_to_save)
