import datetime
import numpy as np
import pickle
from rdkit.Chem.rdchem import BondType

from examples.molecule_search.experiment import get_methane, get_all_mol_metrics
from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from examples.molecule_search.mol_mutations import CHEMICAL_MUTATIONS
from golem.api.main import GOLEM
from golem.core.dag.verification_rules import has_no_isolated_components, has_no_self_cycled_nodes, \
    has_no_isolated_nodes
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimizer import GraphGenerationParams


def test_specifying_parameters_through_api():
    """ Tests that parameters for optimizer are specified correctly. """

    metrics = ['qed_score']

    atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
    all_metrics = get_all_mol_metrics()
    objective = Objective(
        quality_metrics={metric_name: all_metrics[metric_name] for metric_name in metrics},
        is_multi_objective=len(metrics) > 1
    )
    max_heavy_atoms = 50
    pop_size = 10
    num_of_generations = 5
    bond_types = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE)
    adaptive_kind = MutationAgentTypeEnum.bandit
    timeout = datetime.timedelta(minutes=4)

    initial_graphs = [get_methane()]
    initial_graphs = MolAdapter().adapt(initial_graphs)

    golem = GOLEM(
        n_jobs=1,
        timeout=4,
        objective=objective,
        optimizer=EvoGraphOptimizer,
        initial_graphs=initial_graphs,
        pop_size=pop_size,
        max_pop_size=pop_size,
        multi_objective=True,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        elitism_type=ElitismTypesEnum.replace_worst,
        mutation_types=CHEMICAL_MUTATIONS,
        crossover_types=[CrossoverTypesEnum.none],
        adaptive_mutation_type=adaptive_kind,
        adapter=MolAdapter(),
        rules_for_constraint=[has_no_self_cycled_nodes, has_no_isolated_components, has_no_isolated_nodes],
        advisor=MolChangeAdvisor(),
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
        bond_types=bond_types,
        early_stopping_timeout=np.inf,
        early_stopping_iterations=np.inf,
        keep_n_best=4,
        num_of_generations=5,
        keep_history=True,
        history_dir=None,
    )

    # if specify each param class without API
    requirements = MolGraphRequirements(
        max_heavy_atoms=max_heavy_atoms,
        available_atom_types=atom_types or ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
        bond_types=bond_types,
        early_stopping_timeout=np.inf,
        early_stopping_iterations=np.inf,
        keep_n_best=4,
        timeout=timeout,
        num_of_generations=num_of_generations,
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

    assert golem.gp_algorithm_parameters == gp_params
    # compared by pickle dump since there are lots of inner classes with not implemented __eq__ magic methods
    # probably needs to be fixed
    assert pickle.dumps(golem.graph_generation_parameters) == pickle.dumps(graph_gen_params)
    # need to be compared by dicts since the classes itself are different
    assert golem.graph_requirements.__dict__ == requirements.__dict__
