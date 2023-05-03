import os.path
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import Type, Optional, Sequence, List, Iterable, Callable

import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import add_atom, delete_atom, replace_atom, replace_bond, delete_bond, \
    cut_atom, insert_carbon, remove_group, move_group
from examples.molecule_search.mol_metrics import normalized_sa_score, cl_score, penalised_logp, qed_score, \
    normalized_logp
from golem.core.dag.verification_rules import has_no_self_cycled_nodes, has_no_isolated_components, \
    has_no_isolated_nodes
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.visualisation.opt_viz import PlotTypesEnum, OptHistoryVisualizer
from golem.visualisation.opt_viz_extra import visualise_pareto


def get_methane():
    methane = 'C'
    return MolGraph.from_smiles(methane)


def get_all_mol_metrics():
    metrics = {'qed_score': qed_score,
               'cl_score': cl_score,
               'norm_sa_score': normalized_sa_score,
               'penalised_logp': penalised_logp,
               'norm_log_p': normalized_logp}
    return metrics


def molecule_search_setup(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                          max_heavy_atoms: int = 50,
                          atom_types: Optional[List[str]] = None,
                          bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                          timeout: Optional[timedelta] = None,
                          num_iterations: Optional[int] = None,
                          pop_size: int = 20,
                          metrics: Optional[List[str]] = None):
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
        history_dir=os.path.join(os.path.curdir, 'history')
    )
    gp_params = GPAlgorithmParameters(
        pop_size=pop_size,
        max_pop_size=pop_size,
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        elitism_type=ElitismTypesEnum.replace_worst,
        mutation_types=[
            add_atom,
            delete_atom,
            replace_atom,
            replace_bond,
            delete_bond,
            cut_atom,
            insert_carbon,
            remove_group,
            move_group
        ],
        crossover_types=[CrossoverTypesEnum.none],
        adaptive_mutation_type=MutationAgentTypeEnum.bandit
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

    initial_graphs = [get_methane()]
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


def visualize_results(molecules: Iterable[MolGraph],
                      objective: Objective,
                      history: OptHistory,
                      save_path: Optional[str] = None):
    save_path = save_path or os.path.join(os.path.curdir, 'visualisations')
    Path(save_path).mkdir(exist_ok=True)

    if objective.is_multi_objective:
        visualise_pareto(history.archive_history[-1], objectives_names=objective.metric_names[:2], folder=save_path)

    visualizer = OptHistoryVisualizer(history)
    visualization = PlotTypesEnum.fitness_line.value(visualizer.history, visualizer.visuals_params)
    visualization.visualize(dpi=100, save_path=os.path.join(save_path, 'fitness_line.png'))

    rw_molecules = [mol.get_rw_molecule() for mol in set(molecules)]
    objectives = [objective.format_fitness(objective(mol)) for mol in set(molecules)]
    image = Draw.MolsToGridImage(rw_molecules,
                                 legends=objectives,
                                 molsPerRow=min(4, len(rw_molecules)),
                                 subImgSize=(1000, 1000),
                                 legendFontSize=50)
    image.show()
    image.save(os.path.join(save_path, 'best_molecules.png'))


def run_experiment(optimizer_setup: Callable,
                   optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                   max_heavy_atoms: int = 50,
                   atom_types: Optional[List[str]] = None,
                   bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                   pop_size: int = 20,
                   metrics: Optional[List[str]] = None,
                   num_trials: int = 1,
                   trial_timeout: Optional[int] = None,
                   trial_iterations: Optional[int] = None,
                   visualize: bool = False
                   ):
    log = StringIO()
    atom_types = atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
    metrics = metrics or ['qed_score']
    trial_results = []
    experiment_id = f'Experiment [metrics={", ".join(metrics)} pop_size={pop_size}]\n'
    for trial in range(num_trials):
        optimizer, objective = optimizer_setup(optimizer_cls,
                                               max_heavy_atoms,
                                               atom_types,
                                               bond_types,
                                               trial_timeout,
                                               trial_iterations,
                                               pop_size,
                                               metrics)
        found_graphs = optimizer.optimise(objective)
        history = optimizer.history
        if visualize:
            molecules = [MolAdapter().restore(graph) for graph in found_graphs]
            save_path = os.path.join(os.path.curdir,
                                     'visualisations',
                                     f'trial_{trial}_pop_size_{pop_size}_{"_".join(metrics)}')
            visualize_results(set(molecules), objective, history, save_path)
        Path("results").mkdir(exist_ok=True)
        history.save(f'./results/trial_{trial}_pop_size_{pop_size}_{"_".join(metrics)}.json')
        trial_results.extend(history.final_choices)

    # Compute mean & std for metrics of trials
    ff = objective.format_fitness
    trial_metrics = np.array([ind.fitness.values for ind in trial_results])
    trial_metrics_mean = trial_metrics.mean(axis=0)
    trial_metrics_std = trial_metrics.std(axis=0)
    print(f'{experiment_id} finished with metrics:\n'
          f'mean={ff(trial_metrics_mean)}\n'
          f' std={ff(trial_metrics_std)}',
          file=log)
    print(log.getvalue())
    return log.getvalue()


if __name__ == '__main__':
    run_experiment(molecule_search_setup,
                   max_heavy_atoms=38,
                   trial_iterations=100,
                   pop_size=100,
                   metrics=['penalised_logp'],
                   visualize=True,
                   num_trials=10)
