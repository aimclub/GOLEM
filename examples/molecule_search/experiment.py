import random
from datetime import timedelta
from typing import Type, Optional, Sequence

from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import add_atom, delete_atom, replace_atom, replace_bond, delete_bond
from examples.molecule_search.molecule_metrics import normalized_sa_score, cl_score
from golem.core.dag.verification_rules import has_no_self_cycled_nodes
from golem.core.log import Log
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams, GraphOptimizer
from golem.visualisation.opt_history.fitness_box import FitnessBox
from golem.visualisation.opt_history.fitness_line import FitnessLine
from golem.visualisation.opt_viz_extra import visualise_pareto


def load_init_population(path = 'C:\\Users\\admin\\PycharmProjects\GOLEM\examples\molecule_search\data\shingles\guacamol_v1_all.smiles', objective=None):
    with open(path, "r") as f:
        smiles_list = random.sample(f.readlines(), 100)
        print(smiles_list)
    init_pop = [MolGraph.from_smiles(smile) for smile in smiles_list]
    return init_pop


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
        keep_history=True,
        n_jobs=-1
    )
    gp_params = GPAlgorithmParameters(
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.parameter_free,
        mutation_types=[
            add_atom, delete_atom, replace_atom, replace_bond, delete_bond
        ],
        crossover_types=[CrossoverTypesEnum.none]
    )
    graph_gen_params = GraphGenerationParams(
        adapter=MolAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes],
        available_node_types=atom_types,
    )

    objective = Objective(
        quality_metrics={
            'norm_sa_score': normalized_sa_score,
            'cl_score': cl_score
        },
        is_multi_objective=True
    )

    # Generate simple initial population with line graphs
    initial_graphs = load_init_population()
    initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

    # Build the optimizer
    optimiser = optimizer_cls(objective, initial_graphs, requirements, graph_gen_params, gp_params)
    return optimiser, objective


if __name__ == '__main__':
    Log().reset_logging_level(20)
    optimizer, objective = molecule_search_setup(num_iterations=200)
    found_graphs = optimizer.optimise(objective)
    molecules = [MolAdapter().restore(graph) for graph in found_graphs]
    visualise_pareto(optimizer.best_individuals, objectives_names=objective.metric_names)
    FitnessLine(optimizer.history).visualize()
    FitnessBox(optimizer.history).visualize()
