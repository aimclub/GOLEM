import os.path
import random
from datetime import timedelta
from typing import Type, Optional, Sequence, List, Iterable

from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import add_atom, delete_atom, replace_atom, replace_bond, delete_bond, \
    cut_atom, insert_carbon, remove_group, move_group
from examples.molecule_search.molecule_metrics import normalized_sa_score, cl_score, penalised_logp, qed_score
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.opt_history_objects.opt_history import OptHistory
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.visualisation.opt_viz import PlotTypesEnum, OptHistoryVisualizer
from golem.visualisation.opt_viz_extra import visualise_pareto


def load_init_population(path=".\\data\\shingles\\guacamol_v1_all.smiles", objective=None):
    with open(path, "r") as f:
        smiles_list = random.sample(f.readlines(), 100)
        print(smiles_list)
    init_pop = [MolGraph.from_smiles(smile) for smile in smiles_list]
    return init_pop


def molecule_search_setup(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                          max_heavy_atoms: int = 50,
                          atom_types: Optional[List[str]] = None,
                          bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                          timeout: Optional[timedelta] = None,
                          num_iterations: Optional[int] = None):
    requirements = MolGraphRequirements(
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
        bond_types=bond_types,
        early_stopping_timeout=50,
        early_stopping_iterations=1000,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        keep_history=True,
        n_jobs=-1
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
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
        rules_for_constraint=[has_no_self_cycled_nodes],
        advisor=MolChangeAdvisor(),
    )

    objective = Objective(
        quality_metrics={
            'qed_score': qed_score,
            'cl_score': cl_score,
            'norm_sa_score': normalized_sa_score,
            'penalised_logp': penalised_logp,
        },
        is_multi_objective=True
    )

    initial_graphs = load_init_population()
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


def visualize(molecules: Iterable[MolGraph], history: OptHistory, metric_names: Sequence[str]):
    save_path = os.path.join(os.path.curdir, 'visualisations')

    visualise_pareto(history.archive_history[-1], objectives_names=metric_names, folder=save_path)

    for plot_type in [PlotTypesEnum.fitness_line, PlotTypesEnum.fitness_box]:
        visualizer = OptHistoryVisualizer(history)
        visualization = plot_type.value(visualizer.history, visualizer.visuals_params)
        visualization.visualize(dpi=100, save_path=os.path.join(save_path, f'{plot_type.name}.png'))

    rw_molecules = [mol.get_rw_molecule() for mol in set(molecules)]
    image = Draw.MolsToGridImage(rw_molecules, molsPerRow=min(4, len(rw_molecules)), subImgSize=(1000, 1000))
    image.show()
    image.save(os.path.join(save_path, 'best_molecules.png'))


if __name__ == '__main__':
    optimizer, objective = molecule_search_setup(timeout=timedelta(minutes=5))
    found_graphs = optimizer.optimise(objective)
    molecules = [MolAdapter().restore(graph) for graph in found_graphs]
    visualize(molecules, optimizer.history, metric_names=objective.metric_names[:2])
