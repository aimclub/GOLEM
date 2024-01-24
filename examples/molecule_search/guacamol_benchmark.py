import json
import os
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from joblib import delayed
from rdkit.Chem import Draw, MolFromSmiles
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import CHEMICAL_MUTATIONS
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
                         path=".\\data\\guacamol_v1_all.smiles",
                         number_of_molecules=100):
    """ Original code:
     https://github.com/BenevolentAI/guacamol_baselines/blob/master/graph_ga/goal_directed_generation.py"""
    with open(path, "r") as f:
        smiles_list = f.readlines()
    joblist = [delayed(scoring_function.score)(smile) for smile in smiles_list]
    scores = joblib.Parallel(n_jobs=n_jobs)(joblist)
    scored_smiles = list(zip(scores, smiles_list))
    scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
    best_smiles = [smile for score, smile in scored_smiles][:number_of_molecules]
    init_pop = [MolGraph.from_smiles(smile) for smile in best_smiles]
    return init_pop


class GolemMoleculeGenerator(GoalDirectedGenerator):
    """ You need to download Guacamol all smiles dataset from https://figshare.com/projects/GuacaMol/56639"""
    def __init__(self,
                 requirements: Optional[MolGraphRequirements] = None,
                 graph_gen_params: Optional[GraphGenerationParams] = None,
                 gp_params: Optional[GPAlgorithmParameters] = None,
                 trial: int = 0):
        self.requirements = requirements or MolGraphRequirements(
            max_heavy_atoms=50,
            available_atom_types=['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
            bond_types=(BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
            early_stopping_timeout=np.inf,
            early_stopping_iterations=700,
            keep_n_best=4,
            timeout=None,
            num_of_generations=3000,
            keep_history=True,
            n_jobs=-1,
            history_dir=None)

        self.graph_gen_params = graph_gen_params or GraphGenerationParams(
            adapter=MolAdapter(),
            advisor=MolChangeAdvisor())

        self.gp_params = gp_params or GPAlgorithmParameters(
            pop_size=50,
            max_pop_size=50,
            multi_objective=False,
            genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
            elitism_type=ElitismTypesEnum.replace_worst,
            mutation_types=CHEMICAL_MUTATIONS,
            crossover_types=[CrossoverTypesEnum.none],
            adaptive_mutation_type=MutationAgentTypeEnum.bandit)
        self.trial = trial
        self.task_num = 0

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        objective = Objective(
            quality_metrics=lambda mol: -scoring_function.score(mol.get_smiles(aromatic=True)),
            is_multi_objective=False
        )
        self.gp_params.pop_size = max(number_molecules, 100)
        self.gp_params.max_pop_size = self.gp_params.pop_size

        initial_graphs = load_init_population(scoring_function, number_of_molecules=self.gp_params.pop_size)
        initial_graphs = self.graph_gen_params.adapter.adapt(initial_graphs)

        # Build the optimizer
        optimiser = EvoGraphOptimizer(objective,
                                      initial_graphs,
                                      self.requirements,
                                      self.graph_gen_params,
                                      self.gp_params)
        optimiser.optimise(objective)
        history = optimiser.history
        history.save(os.path.join(os.curdir, f'history_trial_{self.trial}_task{self.task_num}.json'))
        self.task_num += 1

        # Take only the first graph's appearance in history
        individuals \
            = list({hash(self.graph_gen_params.adapter.restore(ind.graph)): ind
                    for gen in history.generations
                    for ind in reversed(list(gen))}.values())

        top_individuals = sorted(individuals,
                                 key=lambda pos_ind: pos_ind.fitness, reverse=True)[:number_molecules]
        top_smiles = [MolAdapter().restore(ind.graph).get_smiles(aromatic=True) for ind in top_individuals]
        return top_smiles


def visualize(path: str):
    with open(path) as json_file:
        results = json.load(json_file)
    print(f"Guacamol version: {results['guacamol_version']} \n"
          f"Benchmark version: {results['benchmark_suite_version']} \n")
    results = results['results']
    for result in results:
        generated_molecules, scores = [[MolFromSmiles(smile) for smile, score in result['optimized_molecules'][:12]],
                                       [round(score, 3) for smile, score in result['optimized_molecules'][:12]]]
        benchmark_name = result['benchmark_name']
        scores = [f"{benchmark_name} : {score}" for score in scores]
        image = Draw.MolsToGridImage(generated_molecules,
                                     legends=scores,
                                     molsPerRow=min(4, len(generated_molecules)),
                                     subImgSize=(1000, 1000),
                                     legendFontSize=50)
        image.show()
        image.save(f'{benchmark_name}_results.png')


def get_launch_statistics(paths: List[str]):
    results = []
    for path in paths:
        with open(path) as json_file:
            results.append(json.load(json_file)['results'])

    column_names = ['benchmark', 'mean', 'std', 'min', 'max', 'mean_time']

    df = pd.DataFrame(columns=column_names)

    for bench_num in range(20):
        benchmark = results[0][bench_num]['benchmark_name']
        scores = []
        time_spent = []
        for result in results:
            scores.append(result[bench_num]['score'])
            time_spent.append(result[bench_num]['execution_time'])
        bench_result = pd.DataFrame(data=[[benchmark,
                                           np.mean(scores),
                                           np.std(scores),
                                           np.min(scores),
                                           np.max(scores),
                                           np.mean(time_spent)]],
                                    columns=column_names)
        df = pd.concat([df, bench_result], ignore_index=True, axis=0)
    pd.set_option('display.max_columns', None)
    print(df)
    df.to_csv('results.csv')


if __name__ == '__main__':
    # one launch takes more than 24h
    for launch in range(1):
        print(f'\nLaunch_num {launch}\n')
        assess_goal_directed_generation(GolemMoleculeGenerator(trial=launch),
                                        benchmark_version='v2',
                                        json_output_file=f'output_goal_directed_{launch}.json')
    visualize('output_goal_directed_1.json')
    get_launch_statistics([f'output_goal_directed_{launch}.json' for launch in range(4)])
