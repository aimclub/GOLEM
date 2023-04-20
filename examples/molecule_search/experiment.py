from datetime import timedelta
from typing import Type, Optional, Sequence

from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import add_atom, delete_atom, replace_atom, replace_bond, delete_bond
from examples.molecule_search.molecule_metrics import normalized_sa_score, cl_score
from examples.synthetic_graph_evolution.experiment import run_experiments
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
import random


def load_init_population(path = 'C:\\Users\\admin\\PycharmProjects\GOLEM\examples\molecule_search\data\shingles\guacamol_v1_all.smiles', objective=None):
    with open(path, "r") as f:
        smiles_list = f.readlines()
        print(random.sample(smiles_list, 1))
    init_pop = [MolGraph.from_smiles(smile) for smile in smiles_list]
    for graph in init_pop:
        graph.show()

def molecule_search_setup(optimizer_cls: Type[GraphOptimizer] = EvoGraphOptimizer,
                          max_heavy_atoms: int = 50,
                          atom_types: Sequence[str] = ('C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'),
                          bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                          timeout: Optional[timedelta] = None,
                          num_iterations: Optional[int] = None):
    requirements = MolGraphRequirements(
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types,
        bond_types=bond_types,
        early_stopping_timeout=5,
        early_stopping_iterations=1000,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_iterations,
        n_jobs=1,
        history_dir=None,
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        mutation_types=[
            add_atom, delete_atom, replace_atom, replace_bond, delete_bond
        ],
    )
    graph_gen_params = GraphGenerationParams(
        adapter=MolAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes],
        available_node_types=atom_types,
    )

    # Setup objective that measures some graph-theoretic similarity measure
    objective = Objective(
        quality_metrics={
            'norm_sa_score': normalized_sa_score,
            'cl_score': cl_score
        },
        is_multi_objective=True
    )

    # Generate simple initial population with line graphs
    initial_graphs = [MolGraph.from_smiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")]
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


if __name__ == '__main__':
    # optimizer, objective = molecule_search_setup(timeout=timedelta(minutes=0.5))
    # found_graphs = optimizer.optimise(objective)
    # for graph in found_graphs:
    #     graph.show()
    #     print(objective(graph))
    load_init_population()

