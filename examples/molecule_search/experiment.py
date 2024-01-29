import os.path
from datetime import timedelta
from pathlib import Path
from typing import Type, Optional, Sequence, List, Iterable, Callable, Dict

import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import CHEMICAL_MUTATIONS
from examples.molecule_search.mol_metrics import normalized_sa_score, penalised_logp, qed_score, \
    normalized_logp, CLScorer
from golem.core.dag.verification_rules import has_no_self_cycled_nodes, has_no_isolated_components, \
    has_no_isolated_nodes
from golem.core.optimisers.adaptive.agent_trainer import AgentTrainer
from golem.core.optimisers.adaptive.history_collector import HistoryReader
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.core.paths import project_root
from golem.visualisation.opt_history.multiple_fitness_line import MultipleFitnessLines
from golem.visualisation.opt_viz_extra import visualise_pareto


def get_methane() -> MolGraph:
    methane = 'C'
    return MolGraph.from_smiles(methane)


def get_all_mol_metrics() -> Dict[str, Callable]:
    metrics = {'qed_score': qed_score,
               'cl_score': CLScorer(),
               'norm_sa_score': normalized_sa_score,
               'penalised_logp': penalised_logp,
               'norm_log_p': normalized_logp}
    return metrics


def molecule_search_setup(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                          adaptive_kind: MutationAgentTypeEnum = MutationAgentTypeEnum.random,
                          max_heavy_atoms: int = 50,
                          atom_types: Optional[List[str]] = None,
                          bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                          timeout: Optional[timedelta] = None,
                          num_iterations: Optional[int] = None,
                          pop_size: int = 20,
                          metrics: Optional[List[str]] = None,
                          initial_molecules: Optional[Sequence[MolGraph]] = None):
    requirements = MolGraphRequirements(
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
        bond_types=bond_types,
        early_stopping_timeout=np.inf,
        early_stopping_iterations=np.inf,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        keep_history=True,
        n_jobs=1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        pop_size=pop_size,
        max_pop_size=pop_size,
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
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

    metrics = metrics or ['qed_score']
    all_metrics = get_all_mol_metrics()
    objective = Objective(
        quality_metrics={metric_name: all_metrics[metric_name] for metric_name in metrics},
        is_multi_objective=len(metrics) > 1
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
                   metrics: Optional[List[str]] = None,
                   num_trials: int = 1,
                   trial_timeout: Optional[int] = None,
                   trial_iterations: Optional[int] = None,
                   visualize: bool = False,
                   save_history: bool = True,
                   pretrain_dir: Optional[str] = None,
                   ):
    metrics = metrics or ['qed_score']
    optimizer_id = optimizer_cls.__name__.lower()[:3]
    experiment_id = f'Experiment [optimizer={optimizer_id} metrics={", ".join(metrics)} pop_size={pop_size}]'
    exp_name = f'{optimizer_id}_{adaptive_kind.value}_popsize{pop_size}_min{trial_timeout}_{"_".join(metrics)}'

    atom_types = atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
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
                                               metrics,
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
    run_experiment(molecule_search_setup,
                   adaptive_kind=MutationAgentTypeEnum.bandit,
                   max_heavy_atoms=38,
                   trial_timeout=6,
                   pop_size=50,
                   visualize=True,
                   num_trials=5,
                   # pretrain_dir=os.path.join(project_root(), 'examples', 'molecule_search', 'histories')
                   )
