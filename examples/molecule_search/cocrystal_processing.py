import os.path
import random
import seaborn as sns
from datetime import timedelta
from pathlib import Path
from typing import Type, Optional, Sequence, List, Iterable, Callable, Dict

import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib import pyplot as plt
from rdkit.Chem import Draw, MolFromSmiles
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdchem import RWMol

from examples.molecule_search.experiment import get_methane
from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_metrics import CocrystalsMetrics, sa_score
from examples.molecule_search.mol_mutations import CHEMICAL_MUTATIONS
from golem.core.dag.verification_rules import has_no_self_cycled_nodes, has_no_isolated_components, \
    has_no_isolated_nodes
from golem.core.optimisers.adaptive.agent_trainer import AgentTrainer
from golem.core.optimisers.adaptive.history_collector import HistoryReader
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.archive import ParetoFront
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.core.paths import project_root
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines
from golem.visualisation.opt_viz_extra import visualise_pareto
import sys
from rdkit import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def molecule_search_setup(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                          adaptive_kind: MutationAgentTypeEnum = MutationAgentTypeEnum.random,
                          max_heavy_atoms: int = 50,
                          atom_types: Optional[List[str]] = None,
                          bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                          timeout: Optional[timedelta] = None,
                          num_iterations: Optional[int] = None,
                          pop_size: int = 20,
                          drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2',
                          initial_molecules: Optional[Sequence[MolGraph]] = None):
    requirements = MolGraphRequirements(
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'],
        bond_types=bond_types,
        early_stopping_timeout=np.inf,
        early_stopping_iterations=np.inf,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        keep_history=True,
        n_jobs=-1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        pop_size=pop_size,
        max_pop_size=pop_size,
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.generational,
        elitism_type=ElitismTypesEnum.replace_worst,
        mutation_types=CHEMICAL_MUTATIONS,
        crossover_types=[CrossoverTypesEnum.none],
        adaptive_mutation_type=adaptive_kind,
    )
    graph_gen_params = GraphGenerationParams(
        adapter=MolAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes, has_no_isolated_components, has_no_isolated_nodes],
        advisor=MolChangeAdvisor(),
    )

    metrics = CocrystalsMetrics(drug)
    objective = Objective(
        quality_metrics={'orthogonal_planes': metrics.orthogonal_planes,
                         'unobstructed': metrics.unobstructed,
                         'h_bond_bridging': metrics.h_bond_bridging,
                         'sa_score': sa_score},
        is_multi_objective=True
    )

    initial_graphs = initial_molecules or [get_methane()]
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


def visualize_results(molecules: Iterable[MolGraph],
                      objective: Objective,
                      history: OptHistory,
                      save_path: Path,
                      show: bool = False):
    save_path.mkdir(parents=True, exist_ok=True)

    # Plot pareto front (if multi-objective)
    if objective.is_multi_objective:
        visualise_pareto(history.archive_history[-1],
                         objectives_names=objective.metric_names[:2],
                         folder=str(save_path))

    # Plot fitness convergence
    history.show.fitness_line(dpi=100, save_path=save_path / 'fitness_line.png')
    # Plot diversity
    history.show.diversity_population(save_path=save_path / 'diversity.gif')
    history.show.diversity_line(save_path=save_path / 'diversity_line.png')

    # Plot found molecules
    rw_molecules = [mol.get_rw_molecule() for mol in set(molecules)]
    objectives = [objective.format_fitness(objective(mol)) for mol in set(molecules)]
    image = Draw.MolsToGridImage(rw_molecules,
                                 legends=objectives,
                                 molsPerRow=min(4, len(rw_molecules)),
                                 subImgSize=(1000, 1000),
                                 legendFontSize=50)
    image.save(save_path / 'best_molecules.png')
    if show:
        image.show()


def pretrain_agent(optimizer: EvoGraphOptimizer, objective: Objective, results_dir: str) -> AgentTrainer:
    agent = optimizer.mutation.agent
    trainer = AgentTrainer(objective, optimizer.mutation, agent)
    # load histories
    history_reader = HistoryReader(Path(results_dir))
    # train agent
    trainer.fit(histories=history_reader.load_histories(), validate_each=1)
    return trainer


def run_experiment(optimizer_setup: Callable,
                   optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                   adaptive_kind: MutationAgentTypeEnum = MutationAgentTypeEnum.random,
                   max_heavy_atoms: int = 50,
                   atom_types: Optional[List[str]] = None,
                   bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                   initial_molecules: Optional[Sequence[MolGraph]] = None,
                   pop_size: int = 20,
                   num_trials: int = 1,
                   trial_timeout: Optional[int] = None,
                   trial_iterations: Optional[int] = None,
                   visualize: bool = False,
                   save_history: bool = True,
                   pretrain_dir: Optional[str] = None,
                   drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2'
                   ):
    optimizer_id = optimizer_cls.__name__.lower()[:3]
    experiment_id = f'Experiment [optimizer={optimizer_id} pop_size={pop_size}]'
    exp_name = f'{optimizer_id}_{adaptive_kind.value}_popsize{pop_size}_min{trial_timeout}'

    trial_results = []
    trial_histories = []
    trial_timedelta = timedelta(minutes=trial_timeout) if trial_timeout else None

    for trial in range(num_trials):
        optimizer, objective = optimizer_setup(optimizer_cls,
                                               adaptive_kind,
                                               max_heavy_atoms,
                                               atom_types,
                                               bond_types,
                                               trial_timedelta,
                                               trial_iterations,
                                               pop_size,
                                               drug,
                                               initial_molecules)
        if pretrain_dir:
            pretrain_agent(optimizer, objective, pretrain_dir)

        found_graphs = optimizer.optimise(objective)
        history = optimizer.history

        if visualize:
            molecules = [MolAdapter().restore(graph) for graph in found_graphs]
            save_dir = Path('visualisations') / exp_name / f'trial_{trial}'
            visualize_results(set(molecules), objective, history, save_dir)
        if save_history:
            result_dir = Path('results') / exp_name
            result_dir.mkdir(parents=True, exist_ok=True)
            history.save(result_dir / f'history_trial_{trial}.json')
        trial_results.extend(history.final_choices)
        trial_histories.append(history)

    # Compute mean & std for metrics of trials
    ff = objective.format_fitness
    trial_metrics = np.array([ind.fitness.values for ind in trial_results])
    trial_metrics_mean = trial_metrics.mean(axis=0)
    trial_metrics_std = trial_metrics.std(axis=0)
    print(f'Experiment {experiment_id}\n'
          f'finished with metrics:\n'
          f'mean={ff(trial_metrics_mean)}\n'
          f' std={ff(trial_metrics_std)}')


def plot_experiment_comparison(experiment_ids: Sequence[str], metric_id: int = 0, results_dir='./results'):
    root = Path(results_dir)
    histories = {}
    for exp_name in experiment_ids:
        trials = []
        for history_filename in os.listdir(root / exp_name):
            if history_filename.startswith('history'):
                history = OptHistory.load(root / exp_name / history_filename)
                trials.append(history)
        histories[exp_name] = trials
        print(f'Loaded {len(trials)} trial histories for experiment: {exp_name}')
    # Visualize
    MultipleFitnessLines.from_histories(histories).visualize(metric_id=metric_id)
    return histories


if __name__ == '__main__':
    # initial_smiles = pd.read_csv(
    #     r"C:\Users\admin\PycharmProjects\GOLEM\examples\molecule_search\all_cocrystals_GOLEM_result.csv",
    #     delimiter=',')['0']

    print('loaded')
    initial_smiles = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\generated_mols_VAE_ep3_no_wu_ft6_lr00005_10k_rnd(352).csv")['0']

    adapter = MolAdapter()
    # initial_molecules = []
    # for smiles in initial_smiles:
    #     try:
    #         mol = Individual(adapter.adapt(MolGraph.from_smiles(smiles)))
    #         initial_molecules.append(mol)
    #     except Exception:
    #         continue
    # print('adapted')
    # metrics = CocrystalsMetrics('CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    # objective = Objective(
    #     quality_metrics={'orthogonal_planes': metrics.orthogonal_planes,
    #                      'unobstructed': metrics.unobstructed,
    #                      'h_bond_bridging': metrics.h_bond_bridging,
    #                      'sa_score': sa_score},
    #     is_multi_objective=True)
    # evaluator = MultiprocessingDispatcher(adapter=adapter, n_jobs=-1).dispatch(objective)

    # initial_molecules = evaluator(initial_molecules)
    # print('evaluated')
    # pareto_front = ParetoFront(maxsize=128000)
    # pareto_front.update(initial_molecules)
    # best_initial = pareto_front.items
    # print('pareto')
    # best_smiles = {adapter.restore(ind.graph).get_smiles(): ind for ind in best_initial}
    best_smiles = dict()
    # print('Initial pareto: ', pareto_front.items)

    # for ind in initial_molecules:
    #     if ind.fitness.getValues()[0] <= -0.333 and ind.fitness.getValues()[1] <= -0.5 and ind.fitness.getValues()[2] <= 0.5 and ind.fitness.getValues()[3] <= 3:
    #         best_smiles.update({adapter.restore(ind.graph).get_smiles(): ind})
    # for i in range(20):
    #     print(i)
    #     history = OptHistory.load(fr"D:\Лаба\molecule_seacrh\cocrysals_data\results\evo_random_popsize200_min60_from_VAE_2\history_trial_{i}.json")
    #     individuals \
    #         = list({hash(adapter.restore(ind.graph)): ind
    #                 for gen in history.generations
    #                 for ind in reversed(list(gen))}.values())
    #     for ind in individuals:
    #         if ind.fitness.getValues()[0] < -0.332 and ind.fitness.getValues()[1] <= -0.5 and ind.fitness.getValues()[2] <= 0.5 and ind.fitness.getValues()[3] <= 3:
    #             best_smiles.update({adapter.restore(ind.graph).get_smiles(): ind})
    # root = r'D:\Лаба\molecule_seacrh\cocrysals_data\CVAE_all'
    # for file in os.listdir(root):
    #     initial_smiles = pd.read_csv(os.path.join(root, file))['0']
    #     adapter = MolAdapter()
    #     individuals = []
    #     for smiles in initial_smiles:
    #         try:
    #             mol = Individual(adapter.adapt(MolGraph.from_smiles(smiles)))
    #             individuals.append(mol)
    #         except Exception:
    #             continue
    #     evaluator(individuals)
    #     count = 0
    #     for ind in individuals:
    #         if ind.fitness.getValues()[0] < -0.332 and ind.fitness.getValues()[1] <= -0.5 and ind.fitness.getValues()[
    #             2] <= 0.5 and ind.fitness.getValues()[3] <= 3:
    #             count += 1
    #             best_smiles.update({adapter.restore(ind.graph).get_smiles(): ind})
    #     print(count)
    # result = {'drug': ['CN1C2=C(C(=O)N(C1=O)C)NC=N2'] * len(best_smiles), 'generated_coformers': [],
    #           'orthogonal_planes': [], 'unobstructed': [], 'h_bond_bridging': [], 'sa_score': []}
    #
    # for smiles, ind in best_smiles.items():
    #     result['unobstructed'].append(abs(ind.fitness.values[1]))
    #     result['generated_coformers'].append(smiles)
    #     result['orthogonal_planes'].append(abs(ind.fitness.values[0]))
    #     result['h_bond_bridging'].append(1 - ind.fitness.values[2])
    #     result['sa_score'].append(ind.fitness.values[3])
    #
    # df = pd.DataFrame.from_dict(result)
    # print(len(df))
    # df.to_csv('CVAE_all_valid.csv', index=False)
    #
    # initial_selected = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\GAN_selected_proba_(sa_le_3).csv")
    # n = len(initial_selected)
    # print(n)
    # filtered_vae =
    # filtered_golem_gan = all_golem_from_gan[all_golem_from_gan.sa_score <= 3]
    # filtered_golem_vae = all_golem_from_vae[all_golem_from_vae.sa_score <= 3]
    # filtered_golem_methane = all_golem_from_methane[all_golem_from_methane.sa_score <= 3]
    # filtered_vae = all_golem_from_methane[init_vae.sa_score <= 3]
    #
    #
    # from paretoset import paretoset
    # all_vae = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\all_cocrystals_GOLEM_result_from_VAE_2.csv")
    # remaining_mols = all_vae[all_vae.sa_score <=3]
    # collected = 0
    # pareto_fronts = []
    # while collected < n:
    #     mask = paretoset(remaining_mols[['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 'sa_score']],
    #                      sense=["max", "max", "max", "min"])
    #     front = remaining_mols[mask]
    #     remaining_mols = remaining_mols[~mask]
    #     if collected + len(front) > n:
    #         front = front.sample(n - collected)
    #     collected += len(front)
    #     pareto_fronts.append(front)
    #     # print(collected)
    #
    # filtered_golem_vae = pd.concat(pareto_fronts, ignore_index=True)
    # filtered_golem_vae.to_csv('pareto_best_golem_vae_2_sa_3.csv')


    # filtered_golem_methane = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\pareto_best_golem_methane_sa_3.csv")
    # filtered_golem_gan = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\pareto_best_golem_gan_sa_3.csv")
    # filtered_golem_vae = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\pareto_best_golem_vae_2_sa_3.csv")
    # filtered_vae = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\results\VAE_2_selected_proba_(sa_le_3).csv")
    #
    # #
    # print('vae', len(filtered_vae))
    # print('gan', len(filtered_golem_gan))
    # print('vae', len(filtered_golem_vae))
    # print('methane', len(filtered_golem_methane))
    # #
    # results_df = pd.DataFrame(data={'unobstructed_gan_init': initial_selected.unobstructed,
    #                                 'unobstructed_evo': filtered_golem_methane.unobstructed,
    #                                 'unobstructed_vae': filtered_golem_vae.unobstructed,
    #                                 'unobstructed_vae_init': filtered_vae.unobstructed,
    #                                 'unobstructed_gan': filtered_golem_gan.unobstructed,
    #                                 'orthogonal_planes_gan_init': initial_selected.orthogonal_planes,
    #                                 'orthogonal_planes_evo': filtered_golem_methane.orthogonal_planes,
    #                                 'orthogonal_planes_vae_init': filtered_vae.orthogonal_planes,
    #                                 'orthogonal_planes_vae': filtered_golem_vae.orthogonal_planes,
    #                                 'orthogonal_planes_gan': filtered_golem_gan.orthogonal_planes,
    #                                 'h_bond_bridging_gan_init': initial_selected.h_bond_bridging,
    #                                 'h_bond_bridging_evo': filtered_golem_methane.h_bond_bridging,
    #                                 'h_bond_bridging_vae_init': filtered_vae.h_bond_bridging,
    #                                 'h_bond_bridging_vae': filtered_golem_vae.h_bond_bridging,
    #                                 'h_bond_bridging_gan': filtered_golem_gan.h_bond_bridging,
    #                                 'sa_gan_init': initial_selected.sa_score,
    #                                 'sa_evo': filtered_golem_methane.sa_score,
    #                                 'sa_vae': filtered_golem_vae.sa_score,
    #                                 'sa_vae_init': filtered_vae.sa_score,
    #                                 'sa_gan': filtered_golem_gan.sa_score,
    #                                 })
    #
    # sns.set_theme()
    # sns.set(font_scale=2.5)
    # plt.figure(figsize=(15, 8))
    # my_pal = {"unobstructed_evo": "darkcyan", "unobstructed_gan_init": 'lightseagreen', "unobstructed_gan": 'paleturquoise',
    #           "unobstructed_vae_init": 'lightcyan', "unobstructed_vae": "aliceblue",
    #           "orthogonal_planes_evo": "darkcyan", "orthogonal_planes_gan_init": 'lightseagreen', "orthogonal_planes_gan": 'paleturquoise',
    #           "orthogonal_planes_vae_init": 'lightcyan', "orthogonal_planes_vae": "aliceblue",
    #           "h_bond_bridging_evo": "darkcyan", "h_bond_bridging_gan_init": 'lightseagreen', "h_bond_bridging_gan": 'paleturquoise',
    #           "h_bond_bridging_vae_init": 'lightcyan', "h_bond_bridging_vae": "aliceblue",
    #           "sa_evo": "darkcyan", "sa_gan_init": 'lightseagreen',
    #           "sa_gan": 'paleturquoise',
    #           "sa_vae_init": 'lightcyan', "sa_vae": "aliceblue"
    #           }
    # sns.violinplot(results_df[['unobstructed_evo', 'unobstructed_gan_init', 'unobstructed_gan', 'unobstructed_vae_init', 'unobstructed_vae']],
    #                palette=my_pal)
    # # init_patch = mpatches.Patch(color='powderblue', label='GAN Generated')
    # # evo_patch = mpatches.Patch(color='cadetblue', label='GOLEM optimized from methane')
    # # gan_patch = mpatches.Patch(color='paleturquoise', label='GOLEM optimized from GAN')
    # # vae_patch = mpatches.Patch(color='aliceblue', label='GOLEM optimized from VAE')
    # # plt.legend(handles=[gan_patch, evo_patch, gan_patch, vae_patch])
    # plt.xticks([0, 1, 2, 3, 4], ["GOLEM", "GAN", "GOLEM + GAN", "VAE", "GOLEM + VAE"])
    # plt.title('Unobstructed')
    # plt.savefig("violins_unobstructed_sa_3.png", dpi=500)
    # plt.show()
    #
    # sns.set_theme()
    # sns.set(font_scale=2.5)
    # plt.figure(figsize=(15, 8))
    # sns.violinplot(results_df[['orthogonal_planes_evo', 'orthogonal_planes_gan_init', 'orthogonal_planes_gan', 'orthogonal_planes_vae_init',
    #                            'orthogonal_planes_vae']],
    #                palette=my_pal)
    # plt.xticks([0, 1, 2, 3, 4], ["GOLEM", "GAN", "GOLEM + GAN", "VAE", "GOLEM + VAE"])
    # plt.title('Orthogonal planes')
    # plt.savefig("violins_orthogonal_planes_sa_3.png", dpi=500)
    # plt.show()
    #
    # sns.set_theme()
    # sns.set(font_scale=2.5)
    # plt.figure(figsize=(15, 8))
    # sns.violinplot(results_df[['h_bond_bridging_evo', 'h_bond_bridging_gan_init', 'h_bond_bridging_gan',
    #                            'h_bond_bridging_vae_init',
    #                            'h_bond_bridging_vae']],
    #                palette=my_pal)
    # plt.xticks([0, 1, 2, 3, 4], ["GOLEM", "GAN", "GOLEM + GAN", "VAE", "GOLEM + VAE"])
    # plt.title('H-bond bridging')
    # plt.savefig("violins_h_bond_bridging_sa_3.png", dpi=500)
    # plt.show()
    #
    # sns.set_theme()
    # sns.set(font_scale=2.5)
    # plt.figure(figsize=(15, 8))
    # sns.violinplot(results_df[['sa_evo', 'sa_gan_init', 'sa_gan',
    #                            'sa_vae_init',
    #                            'sa_vae']],
    #                palette=my_pal)
    # plt.xticks([0, 1, 2, 3, 4], ["GOLEM", "GAN", "GOLEM + GAN", "VAE", "GOLEM + VAE"])
    # plt.title('sa')
    # plt.savefig("violins_sa_sa_3.png", dpi=500)
    # plt.show()
    #
    # pd.set_option('display.max_columns', None)
    # print(results_df[['unobstructed_evo', 'unobstructed_gan_init', 'unobstructed_gan', 'unobstructed_vae_init', 'unobstructed_vae']].describe().T)
    #
    # pd.set_option('display.max_columns', None)
    # print(results_df[['h_bond_bridging_evo', 'h_bond_bridging_gan_init', 'h_bond_bridging_gan',
    #                            'h_bond_bridging_vae_init',
    #                            'h_bond_bridging_vae']].describe().T)
    #
    # pd.set_option('display.max_columns', None)
    # print(results_df[['orthogonal_planes_evo', 'orthogonal_planes_gan_init', 'orthogonal_planes_gan', 'orthogonal_planes_vae_init',
    #                            'orthogonal_planes_vae']].describe().T)
    # # import scipy.stats as stats
    # # print('unobstructed', initial_selected.unobstructed.median(), filtered_golem.unobstructed.median())
    # # print('orthogonal_planes', initial_selected.orthogonal_planes.median(), filtered_golem.orthogonal_planes.median())
    # # print('h_bond_bridging', initial_selected.h_bond_bridging.median(), filtered_golem.h_bond_bridging.median())
    # #
    # # # perform two-sided test. You can use 'greater' or 'less' for one-sided test
    # # res = stats.mannwhitneyu(x=initial_selected.unobstructed, y=filtered_golem.unobstructed, alternative='less')
    # # print('unobstructed', res)
    # # res = stats.mannwhitneyu(x=initial_selected.orthogonal_planes, y=filtered_golem.orthogonal_planes,
    # #                          alternative='less')
    # # print('orthogonal_planes', res)
    # #
    # # res = stats.mannwhitneyu(x=initial_selected.h_bond_bridging, y=filtered_golem.h_bond_bridging,
    # #                          alternative='less')
    # #
    # # print('h_bond_bridging', res)
    # #
    # # sa_score_data = pd.DataFrame(data={'sa_score_gan': initial_selected.sa_score,
    # #                                    'sa_score_evo': filtered_golem.sa_score})
    # # sns.violinplot(sa_score_data)
    # # plt.show()
    #
    # # molecules = [MolGraph.from_smiles(mol) for mol in filtered_golem.generated_coformers.sample(12)]
    # # rw_molecules = [mol.get_rw_molecule() for mol in molecules]
    # # metrics = CocrystalsMetrics('CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    # # objective = Objective(
    # #     quality_metrics={'orth_pl': metrics.orthogonal_planes,
    # #                      'unobstr': metrics.unobstructed,
    # #                      'hbb': metrics.h_bond_bridging,
    # #                      'sa': sa_score},
    # #     is_multi_objective=True
    # # )
    # # objectives = [objective.format_fitness(objective(mol)) for mol in set(molecules)]
    # # image = Draw.MolsToGridImage(rw_molecules,
    # #                              legends=objectives,
    #                              molsPerRow=min(4, len(rw_molecules)),
    #                              subImgSize=(1000, 1000),
    #                              legendFontSize=50)
    # # image.save(r'D:\Лаба\molecule_seacrh\cocrysals_data\pareto_best_molecules_golem_max_sa_3.png')
    # image.show()
    #
    # #

    # initial_molecules = [get_methane()]
    # run_experiment(molecule_search_setup,
    #                adaptive_kind=MutationAgentTypeEnum.random,
    #                initial_molecules=initial_molecules,
    #                max_heavy_atoms=50,
    #                trial_timeout=60,
    #                trial_iterations=200,
    #                pop_size=200,
    #                visualize=False,
    #                num_trials=20,
    #                drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2'
    #                )
