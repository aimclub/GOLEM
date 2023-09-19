import os

from scipy.stats import mannwhitneyu

from experiments.experiment_analyzer import ExperimentAnalyzer
from golem.core.paths import project_root
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines

if __name__ == '__main__':
    path_to_root = os.path.join('Z:\\', 'Pinchuk', 'golem_vs_bandit')

    base_path = os.path.join(path_to_root, 'test_results')

    path_to_save = os.path.join(path_to_root, 'test_result_analysis')

    metrics = ['sp_adj', 'sp_lapl', 'graph_size', 'degree']

    analyzer = ExperimentAnalyzer(path_to_root=base_path, folders_to_ignore=['result_analysis',
                                                                             'Thumbs.db'])

    analyzer.plot_convergence(path_to_save=os.path.join(path_to_save, 'convergence_plots'),
                              history_folder=None, metrics_names=metrics)
    #
    # analyzer.analyze_metrics(metric_names=metrics,
    #                          file_name='metrics.csv', is_mean=True,
    #                          keen_n_best=4,
    #                          path_to_save=path_to_save)
    #
    # metrics = analyzer.analyze_metrics(metric_names=metrics,
    #                                    file_name='metrics.csv', is_mean=False,
    #                                    keen_n_best=4)

    # for metric in metrics.keys():
    #     analyzer.analyze_statistical_significance(stat_tests=[mannwhitneyu],
    #                                               data_to_analyze=metrics[metric], path_to_save=path_to_save)
    #
    # line = MultipleFitnessLines.from_saved_histories(root_path=base_path)
    # line.visualize(path_to_save=os.path.join(path_to_save, 'convergence_plots'), metric_id=0)
    # line.visualize(path_to_save=os.path.join(path_to_save, 'convergence_plots'), metric_id=1)
    # line.visualize(path_to_save=os.path.join(path_to_save, 'convergence_plots'), metric_id=2)
    # line.visualize(path_to_save=os.path.join(path_to_save, 'convergence_plots'), metric_id=3)

    #
    # convergence = analyzer.analyze_convergence(history_folder=None, is_mean=True, path_to_save=path_to_save)
    #
    # analyzer.plot_convergence(path_to_save=os.path.join(path_to_save, 'plot_convergence'))
    #
    # for metric in convergence.keys():
    #     analyzer.analyze_statistical_significance(stat_tests=[mannwhitneyu],
    #                                               data_to_analyze=convergence[metric], path_to_save=path_to_save)
