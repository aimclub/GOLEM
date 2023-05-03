import os
import random
from typing import Optional, List

import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import add_atom, delete_atom, replace_atom, delete_bond, replace_bond, \
    cut_atom, insert_carbon, remove_group, move_group
from golem.core.dag.verification_rules import has_no_self_cycled_nodes, has_no_isolated_components, \
    has_no_isolated_nodes
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams


def load_init_population(path=".\\data\\shingles\\guacamol_v1_all.smiles", pop_size=20, objective=None):
    with open(path, "r") as f:
        smiles_list = random.sample(f.readlines(), pop_size)
    init_pop = [MolGraph.from_smiles(smile) for smile in smiles_list]
    return init_pop


class GolemMoleculeGenerator(GoalDirectedGenerator):
    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        requirements = MolGraphRequirements(
            max_heavy_atoms=50,
            available_atom_types=['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
            bond_types=(BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
            early_stopping_timeout=np.inf,
            early_stopping_iterations=np.inf,
            keep_n_best=4,
            timeout=None,
            num_of_generations=10,
            keep_history=True,
            n_jobs=1,
            history_dir=os.path.join(os.path.curdir, 'guacamol_history')
        )
        gp_params = GPAlgorithmParameters(
            pop_size=2,
            max_pop_size=2,
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

        objective = Objective(
            quality_metrics=lambda mol: -scoring_function.score(mol.get_smiles(aromatic=True)),
            is_multi_objective=False
        )

        initial_graphs = load_init_population()
        initial_graphs = graph_gen_params.adapter.adapt(initial_graphs)

        # Build the optimizer
        optimiser = EvoGraphOptimizer(objective, initial_graphs, requirements, graph_gen_params, gp_params)
        optimiser.optimise(objective)
        history = optimiser.history

        # Take only the first graph's appearance in history
        individuals_with_positions \
            = list({ind.graph.descriptive_id: ind
                    for gen in history.individuals
                    for ind in reversed(list(gen))}.values())

        top_individuals = sorted(individuals_with_positions,
                                 key=lambda pos_ind: pos_ind.fitness, reverse=True)[:number_molecules]
        top_smiles = [MolAdapter().restore(ind.graph).get_smiles(aromatic=True) for ind in top_individuals]
        return top_smiles


if __name__ == '__main__':
    assess_goal_directed_generation(GolemMoleculeGenerator())
