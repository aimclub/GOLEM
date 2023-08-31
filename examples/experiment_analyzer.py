import os

from experiments.experiment_analyzer import ExperimentAnalyzer

if __name__ == '__main__':
    path_to_root = os.path.join('Z:\Pinchuk\MetaAutoML')

    path_to_save = os.path.join(path_to_root, 'result_analysis')
    analyzer = ExperimentAnalyzer(path_to_root=path_to_root, folders_to_ignore=['result_analysis',
                                                                                'Thumbs.bd'])
    convergence = analyzer.analyze_convergence(history_folder='histories', is_raise=False,
                                               path_to_save=path_to_save)

    metrics_dict = analyzer.analyze_metrics(metric_names=['roc_auc'], file_name='evaluation_results.csv',
                                            is_raise=False, path_to_save=path_to_save)
    print(metrics_dict)

    # path_to_save = os.path.join('Z:\Pinchuk')
    # if not os.path.exists(path_to_save):
    #     os.makedirs(path_to_save)
    # analyzer.plot_convergence(history_folder='histories', path_to_save=path_to_root)