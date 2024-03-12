import os
import scipy.stats as stats
from pathlib import Path
from typing import List, Optional

import pandas as pd
import seaborn as sns
from rdkit.Chem import Draw, MolFromSmiles

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_graph import MolGraph
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines
from matplotlib import pyplot as plt


def get_final_dataset(results_folder):
    best_smiles = dict()
    adapter = MolAdapter()

    for file in os.listdir(results_folder):
        if file.startswith('history'):
            history = OptHistory.load(os.path.join(results_folder, file))

            individuals \
                = list({hash(adapter.restore(ind.graph)): ind
                        for gen in history.generations
                        for ind in reversed(list(gen))}.values())
            for ind in individuals:
                if ind.fitness.getValues()[0] <= -0.332 and ind.fitness.getValues()[1] <= -0.5 and \
                        ind.fitness.getValues()[2] <= 0.5 and ind.fitness.getValues()[3] <= 3:
                    best_smiles.update({adapter.restore(ind.graph).get_smiles(): ind})
    result = {'drug': ['CN1C2=C(C(=O)N(C1=O)C)NC=N2'] * len(best_smiles), 'generated_coformers': [],
              'orthogonal_planes': [], 'unobstructed': [], 'h_bond_bridging': [], 'sa_score': []}

    for smiles, ind in best_smiles.items():
        result['unobstructed'].append(abs(ind.fitness.values[1]))
        result['generated_coformers'].append(smiles)
        result['orthogonal_planes'].append(abs(ind.fitness.values[0]))
        result['h_bond_bridging'].append(1 - ind.fitness.values[2])
        result['sa_score'].append(ind.fitness.values[3])

    df = pd.DataFrame.from_dict(result)
    df.to_csv(os.path.join(results_folder, 'all_valid.csv'), index=False)
    return df


def visualize_results(feature_names, initial_molecules, golem_result, results_folder, objective):
    os.makedirs(os.path.join(results_folder, 'visualisations'), exist_ok=True)
    for feature_name in feature_names:
        my_pal = {"initial": "darkcyan", "evo": 'paleturquoise'}
        df = pd.DataFrame(data={'initial': initial_molecules[feature_name],
                                'evo': golem_result[feature_name]
                                })
        sns.violinplot(df, palette=my_pal)
        pd.set_option('display.max_columns', None)
        print(feature_name)
        statistics = df.describe().T
        print(statistics)
        statistics.to_csv(os.path.join(results_folder, f'{feature_name}_stats.csv'))
        plt.xticks([0, 1], ["Initial", "Evo"])
        plt.title(feature_name)
        plt.savefig(os.path.join(results_folder, 'visualisations', f"violins_{feature_name}.png"), dpi=250)
        plt.close()

    # Plot found molecules
    rw_molecules = [MolFromSmiles(mol) for mol in golem_result['generated_coformers']]
    objectives = [objective.format_fitness(objective(MolGraph.from_smiles(mol)))
                  for mol in golem_result['generated_coformers']]
    image = Draw.MolsToGridImage(rw_molecules,
                                 legends=objectives,
                                 molsPerRow=min(4, len(rw_molecules)),
                                 subImgSize=(1000, 1000),
                                 legendFontSize=50)
    image.save(os.path.join(results_folder, 'best_molecules.png'))


def plot_experiment_comparison(experiment_ids: List[str],
                               metric_ids: Optional[List[int]] = None,
                               results_dir='./results'):
    mlp_line = MultipleFitnessLines.from_saved_histories(experiment_ids, root_path=Path(results_dir))

    metric_ids = metric_ids or [0]
    for metric_id in metric_ids:
        mlp_line.visualize(metric_id=metric_id, with_confidence=True, save_path=os.path.join(results_dir, f'mlp_line_metric_{metric_id}.png'))


def get_statistical_significance(feature_names, initial_molecules, golem_result, results_folder):
    data = {'feature': [], 'init_median': [], 'evo_median': [], 'significant': [], 'statistic': [], 'pvalue': []}
    for feature in feature_names:
        stat_sign = False
        res = stats.mannwhitneyu(x=initial_molecules[feature], y=golem_result[feature],
                                 alternative='two-sided')
        if res.pvalue < 0.05:
            res = stats.mannwhitneyu(x=initial_molecules[feature], y=golem_result[feature],
                                     alternative='less')
            stat_sign = res.pvalue < 0.05
        data['feature'].append(feature)
        data['init_median'].append(initial_molecules[feature].median())
        data['evo_median'].append(golem_result[feature].median())
        data['significant'].append(stat_sign)
        data['statistic'].append(res.statistic)
        data['pvalue'].append(res.pvalue)
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(results_folder, 'stat_significance.csv'))
