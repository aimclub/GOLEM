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
                          drug = 'CN1C2=C(C(=O)N(C1=O)C)NC=N2',
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
                   drug = 'CN1C2=C(C(=O)N(C1=O)C)NC=N2'
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
    #     delimiter=',').generated_coformers
    #
    # print('loaded')
    # adapter = MolAdapter()
    # initial_molecules = [Individual(adapter.adapt(MolGraph.from_smiles(smiles))) for smiles in initial_smiles]
    # print('adapted')
    # metrics = CocrystalsMetrics('CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    # objective = Objective(
    #     quality_metrics={'orthogonal_planes': metrics.orthogonal_planes,
    #                      'unobstructed': metrics.unobstructed,
    #                      'h_bond_bridging': metrics.h_bond_bridging},
    #     is_multi_objective=True)
    # evaluator = MultiprocessingDispatcher(adapter=adapter, n_jobs=-1).dispatch(objective)
    #
    # initial_molecules = evaluator(initial_molecules)
    # print('evaluated')
    # pareto_front = ParetoFront(maxsize=128000)
    # pareto_front.update(initial_molecules)
    # best_initial = pareto_front.items
    # print('pareto')
    # best_smiles = {adapter.restore(ind.graph).get_smiles(): ind for ind in best_initial}
    # best_smiles = dict()
    # # print('Initial pareto: ', pareto_front.items)
    # for i in range(20):
    #     print(i)
    #     history = OptHistory.load(fr"D:\Лаба\molecule_seacrh\cocrysals_data\results\evo_random_popsize200_min60_from_C\history_trial_{i}.json")
    #     individuals \
    #         = list({hash(adapter.restore(ind.graph)): ind
    #                 for gen in history.generations
    #                 for ind in reversed(list(gen))}.values())
    #     for ind in individuals:
    #         if ind.fitness.getValues()[0] <= -0.332 and ind.fitness.getValues()[1] <= -0.5 and ind.fitness.getValues()[2] <= 0.5:
    #             best_smiles.update({adapter.restore(ind.graph).get_smiles(): ind})
    # result = {'drug': ['CN1C2=C(C(=O)N(C1=O)C)NC=N2'] * len(best_smiles), 'generated_coformers': [],
    #           'orthogonal_planes': [], 'unobstructed': [], 'h_bond_bridging': [], 'sa_score': []}
    # for smiles, ind in best_smiles.items():
    #     result['unobstructed'].append(abs(ind.fitness.values[1]))
    #     result['generated_coformers'].append(smiles)
    #     result['orthogonal_planes'].append(abs(ind.fitness.values[0]))
    #     result['h_bond_bridging'].append(1 - ind.fitness.values[2])
    #     result['sa_score'].append(ind.fitness.values[3])
    #
    # df = pd.DataFrame.from_dict(result)
    # df.to_csv('all_cocrystals_GOLEM_result_from_methane.csv', index=False)


    all_golem = pd.read_csv(r"C:\Users\admin\PycharmProjects\GOLEM\examples\molecule_search\all_cocrystals_GOLEM_result_from_methane.csv")
    initial_selected = pd.read_csv(r"D:\Лаба\molecule_seacrh\cocrysals_data\dataset_selected_proba.csv")
    initial_selected['sa_score'] = [sascorer.calculateScore(MolFromSmiles(smiles)) for smiles in
                                    initial_selected.generated_coformers]
    initial_selected = initial_selected[initial_selected.sa_score <= 3]
    n = len(initial_selected)
    print(n)
    filtered_golem = all_golem[all_golem.sa_score <= 3]
    print(len(filtered_golem))

    from paretoset import paretoset

    remaining_mols = filtered_golem
    collected = 0
    pareto_fronts = []
    while collected < n:
        mask = paretoset(remaining_mols[['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 'sa_score']], sense=["max", "max", "max", "min"])
        front = remaining_mols[mask]
        remaining_mols = remaining_mols[~mask]
        if collected + len(front) > n:
            front = front.sample(n - collected)
        collected += len(front)
        pareto_fronts.append(front)
        print(collected)

    filtered_golem = pd.concat(pareto_fronts, ignore_index=True)
    # filtered_golem.to_csv('pareto_best_golem_methane_sa_3.csv')
    # initial_selected.to_csv('dataset_selected_proba_methane_sa_3.csv')
    #
    #
    # results_df = pd.DataFrame(data={'unobstructed_gan': initial_selected.unobstructed,
    #                                 'unobstructed_evo': filtered_golem.unobstructed,
    #                                 'orthogonal_planes_gan': initial_selected.orthogonal_planes,
    #                                 'orthogonal_planes_evo': filtered_golem.orthogonal_planes,
    #                                 'h_bond_bridging_gan': initial_selected.h_bond_bridging,
    #                                 'h_bond_bridging_evo': filtered_golem.h_bond_bridging})
    # sns.set_theme()
    # sns.set(font_scale=2.5)
    # plt.figure(figsize=(15, 8))
    # my_pal = {"unobstructed_gan": "powderblue", "unobstructed_evo": "cadetblue",
    #           "orthogonal_planes_gan": "powderblue", "orthogonal_planes_evo": 'cadetblue',
    #           'h_bond_bridging_gan': 'powderblue', 'h_bond_bridging_evo': 'cadetblue'}
    # sns.violinplot(results_df, palette=my_pal)
    # gan_patch = mpatches.Patch(color='powderblue', label='Generated')
    # evo_patch = mpatches.Patch(color='cadetblue', label='Optimized')
    # plt.legend(handles=[gan_patch, evo_patch])
    # locs, labels = plt.xticks()  # Get the current locations and labels.
    # print(locs)
    # plt.xticks([0.5, 2.5, 4.5], ["Unobstructed planes", "Orthogonal planes", "H-bond bridging"])
    # plt.savefig("violins_methane_sa_3.png", dpi=500)
    # plt.show()

    import scipy.stats as stats
    print('unobstructed', initial_selected.unobstructed.median(), filtered_golem.unobstructed.median())
    print('orthogonal_planes', initial_selected.orthogonal_planes.median(), filtered_golem.orthogonal_planes.median())
    print('h_bond_bridging', initial_selected.h_bond_bridging.median(), filtered_golem.h_bond_bridging.median())

    # perform two-sided test. You can use 'greater' or 'less' for one-sided test
    res = stats.mannwhitneyu(x=initial_selected.unobstructed, y=filtered_golem.unobstructed, alternative='less')
    print('unobstructed', res)
    res = stats.mannwhitneyu(x=initial_selected.orthogonal_planes, y=filtered_golem.orthogonal_planes,
                             alternative='less')
    print('orthogonal_planes', res)

    res = stats.mannwhitneyu(x=initial_selected.h_bond_bridging, y=filtered_golem.h_bond_bridging,
                             alternative='less')

    print('h_bond_bridging', res)

    sa_score_data = pd.DataFrame(data={'sa_score_gan': initial_selected.sa_score,
                                       'sa_score_evo': filtered_golem.sa_score})
    sns.violinplot(sa_score_data)
    plt.show()

    molecules = [MolGraph.from_smiles(mol) for mol in filtered_golem.generated_coformers.sample(12)]
    rw_molecules = [mol.get_rw_molecule() for mol in molecules]
    metrics = CocrystalsMetrics('CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    objective = Objective(
        quality_metrics={'orth_pl': metrics.orthogonal_planes,
                         'unobstr': metrics.unobstructed,
                         'hbb': metrics.h_bond_bridging,
                         'sa': sa_score},
        is_multi_objective=True
    )
    objectives = [objective.format_fitness(objective(mol)) for mol in set(molecules)]
    image = Draw.MolsToGridImage(rw_molecules,
                                 legends=objectives,
                                 molsPerRow=min(4, len(rw_molecules)),
                                 subImgSize=(1000, 1000),
                                 legendFontSize=50)
    # image.save(r'D:\Лаба\molecule_seacrh\cocrysals_data\pareto_best_molecules_golem_max_sa_3.png')
    image.show()

    #


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
