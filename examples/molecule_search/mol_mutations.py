from random import choice
from typing import Sequence

import numpy as np
from rdkit.Chem.rdchem import Atom, GetPeriodicTable

from examples.molecule_search.constants import SULFUR_DEFAULT_VALENCE
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters


def add_atom(mol_graph: MolGraph,
             requirements: MolGraphRequirements,
             graph_gen_params: GraphGenerationParams = None,
             parameters: AlgorithmParameters = None):
    atoms_to_connect = get_atom_ids_to_connect_to(mol_graph)
    if len(atoms_to_connect) != 0 or mol_graph.heavy_atoms_number < requirements.max_heavy_atoms:
        connect_to_atom_id = int(choice(atoms_to_connect))
        atom_type_to_add = "Br"
        mol_graph.add_atom(atom_type_to_add)
        new_atom_id = mol_graph.heavy_atoms_number - 1
        mol_graph.add_bond(connect_to_atom_id, new_atom_id)
        return mol_graph
    else:
        return mol_graph


def get_atom_ids_to_connect_to(mol_graph: MolGraph) -> Sequence[int]:
    molecule = mol_graph.get_rw_molecule()
    atoms = np.array(molecule.GetAtoms())
    free_electrons_vector = np.array([get_free_electrons_num(atom) for atom in atoms])
    formal_charge_vector = np.array([atom.GetFormalCharge() for atom in atoms])
    atom_ids = np.arange(mol_graph.heavy_atoms_number)
    return atom_ids[(formal_charge_vector == 0) & (free_electrons_vector > 0)]


def get_free_electrons_num(atom: Atom) -> int:
    atom.UpdatePropertyCache()
    atom_type = atom.GetSymbol()
    if atom_type == "S":
        default_valence = SULFUR_DEFAULT_VALENCE
    else:
        default_valence = GetPeriodicTable().GetDefaultValence(atom_type)
    explicit_valence = atom.GetExplicitValence()
    free_electrons = default_valence - explicit_valence
    return free_electrons


def delete_atom(mol_graph: MolGraph,
                requirements: GraphRequirements,
                graph_gen_params: GraphGenerationParams,
                parameters: AlgorithmParameters):
    pass


def replace_atom(mol_graph: MolGraph,
                 requirements: GraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 parameters: AlgorithmParameters,
                 ):
    pass


def replace_bond(mol_graph: MolGraph,
                 requirements: GraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 parameters: AlgorithmParameters,
                 ):
    pass


if __name__ == '__main__':
    graph = MolGraph.from_smiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    graph.show()
    new_graph = add_atom(graph, MolGraphRequirements())
    new_graph.show()
