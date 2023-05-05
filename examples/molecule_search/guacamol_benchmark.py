from typing import Optional, List

import joblib
import numpy as np
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from joblib import delayed
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import add_atom, delete_atom, replace_atom, delete_bond, replace_bond, \
    cut_atom, insert_carbon, remove_group, move_group
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams


def load_init_population(scoring_function: ScoringFunction,
                         n_jobs: int = -1,
                         path=".\\data\\shingles\\guacamol_v1_all.smiles"):
    with open(path, "r") as f:
        smiles_list = f.readlines()
    joblist = [delayed(scoring_function.score)(smile) for smile in smiles_list]
    scores = joblib.Parallel(n_jobs=n_jobs)(joblist)
    scored_smiles = list(zip(scores, smiles_list))
    scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
    best_smiles = [smile for score, smile in scored_smiles][:100]
    init_pop = [MolGraph.from_smiles(smile) for smile in best_smiles]
    return init_pop


class GolemMoleculeGenerator(GoalDirectedGenerator):
    def __init__(self,
                 requirements: Optional[MolGraphRequirements] = None,
                 graph_gen_params: Optional[GraphGenerationParams] = None,
                 gp_params: Optional[GPAlgorithmParameters] = None):
        self.requirements = requirements or MolGraphRequirements(
            max_heavy_atoms=50,
            available_atom_types=['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
            bond_types=(BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
            early_stopping_timeout=np.inf,
            early_stopping_iterations=50,
            keep_n_best=4,
            timeout=None,
            num_of_generations=500,
            keep_history=True,
            n_jobs=-1,
            history_dir=None)

        self.graph_gen_params = graph_gen_params or GraphGenerationParams(
            adapter=MolAdapter(),
            advisor=MolChangeAdvisor())

        self.gp_params = gp_params or GPAlgorithmParameters(
            pop_size=2,
            max_pop_size=2,
            multi_objective=False,
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
            adaptive_mutation_type=MutationAgentTypeEnum.bandit)

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        objective = Objective(
            quality_metrics=lambda mol: -scoring_function.score(mol.get_smiles(aromatic=True)),
            is_multi_objective=False
        )
        self.gp_params.pop_size = max(number_molecules // 2, 100)
        self.gp_params.max_pop_size = min(self.gp_params.pop_size * 10, 1000)

        initial_graphs = load_init_population(scoring_function)
        initial_graphs = self.graph_gen_params.adapter.adapt(initial_graphs)

        # Build the optimizer
        optimiser = EvoGraphOptimizer(objective,
                                      initial_graphs,
                                      self.requirements,
                                      self.graph_gen_params,
                                      self.gp_params)
        optimiser.optimise(objective)
        history = optimiser.history

        # Take only the first graph's appearance in history
        individuals \
            = list({hash(self.graph_gen_params.adapter.restore(ind.graph)): ind
                    for gen in history.individuals
                    for ind in reversed(list(gen))}.values())

        top_individuals = sorted(individuals,
                                 key=lambda pos_ind: pos_ind.fitness, reverse=True)[:number_molecules]
        top_smiles = [MolAdapter().restore(ind.graph).get_smiles(aromatic=True) for ind in top_individuals]
        return top_smiles


if __name__ == '__main__':
    for launch in range(10):
        print(f'\nLaunch_num {launch}\n')
        assess_goal_directed_generation(GolemMoleculeGenerator(),
                                        benchmark_version='v2',
                                        json_output_file=f'output_goal_directed_{launch}.json')
