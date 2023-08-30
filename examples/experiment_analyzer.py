import os

from experiments.ExperimentAnalyzer import ExperimentAnalyzer

if __name__ == '__main__':
    path_to_root = os.path.join('Z:\Pinchuk\MetaAutoML')
    analyzer = ExperimentAnalyzer(path_to_root=path_to_root)
    analyzer.analyze_convergence(history_folder='histories')