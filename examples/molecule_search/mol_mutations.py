from copy import deepcopy
from random import choice
from typing import Sequence, Tuple

import networkx as nx
import numpy as np
from rdkit.Chem.rdchem import Atom, GetPeriodicTable, Bond

from examples.molecule_search.constants import SULFUR_DEFAULT_VALENCE
from examples.molecule_search.mol_advisor import MolChangeAdvisor
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters


def add_atom(mol_graph: MolGraph,
             requirements: MolGraphRequirements,
             graph_gen_params: GraphGenerationParams = None,
             parameters: AlgorithmParameters = None):
    mol_graph = deepcopy(mol_graph)
    atoms_to_connect = get_atom_ids_to_connect_to(mol_graph)
    if len(atoms_to_connect) != 0 or mol_graph.heavy_atoms_number < requirements.max_heavy_atoms:
        connect_to_atom_id = int(choice(atoms_to_connect))
        connect_to_atom = mol_graph.get_rw_molecule().GetAtomWithIdx(connect_to_atom_id)

        atom_type_to_add = MolChangeAdvisor.propose_atom_type(connect_to_atom, requirements.available_operations)

        mol_graph.add_atom(atom_type_to_add)

        new_atom_id = mol_graph.heavy_atoms_number - 1
        mol_graph.set_bond(connect_to_atom_id, new_atom_id)
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
                graph_gen_params: GraphGenerationParams = None,
                parameters: AlgorithmParameters = None):
    mol_graph = deepcopy(mol_graph)
    atoms_to_delete = get_atoms_to_remove(mol_graph)
    atom_to_delete_id = int(choice(atoms_to_delete))
    mol_graph.remove_atom(atom_to_delete_id)
    return mol_graph


def get_atoms_to_remove(mol_graph: MolGraph) -> Sequence[int]:
    nx_graph = mol_graph.get_nx_graph()
    art_points_ids = nx.articulation_points(nx_graph)
    all_atoms_ids = np.arange(mol_graph.heavy_atoms_number)
    return list(set(all_atoms_ids) - set(art_points_ids))


def replace_atom(mol_graph: MolGraph,
                 requirements: GraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 parameters: AlgorithmParameters):
    pass


def replace_bond(mol_graph: MolGraph,
                 requirements: MolGraphRequirements,
                 graph_gen_params: GraphGenerationParams = None,
                 parameters: AlgorithmParameters = None):
    mol_graph = deepcopy(mol_graph)
    molecule = mol_graph.get_rw_molecule()
    atom_pairs_to_connect = get_atom_pairs_to_connect(mol_graph)
    pair_to_connect = choice(atom_pairs_to_connect)

    if pair_to_connect:
        from_atom = molecule.GetAtomWithIdx(pair_to_connect[0])
        to_atom = molecule.GetAtomWithIdx(pair_to_connect[1])
        pair_free_electrons = min(get_free_electrons_num(from_atom), get_free_electrons_num(to_atom))
        current_bond = molecule.GetBondBetweenAtoms(*pair_to_connect)
        if current_bond:
            current_bond_degree = int(Bond.GetBondType(current_bond))
        else:
            current_bond_degree = 0
        max_bond_degree = current_bond_degree + pair_free_electrons
        possible_bonds = [bond for bond in requirements.bond_types if int(bond) <= max_bond_degree]
        if possible_bonds:
            bond_to_replace = choice(possible_bonds)
            mol_graph.set_bond(bond_type=bond_to_replace, *pair_to_connect)

    return mol_graph


def get_atom_pairs_to_connect(mol_graph: MolGraph) -> Sequence[Tuple[int, int]]:
    molecule = mol_graph.get_rw_molecule()
    atoms = np.array(molecule.GetAtoms())
    atom_pairs_to_connect = []
    for from_atom_id, from_atom in enumerate(atoms):
        for to_atom_id, to_atom in enumerate(atoms[from_atom_id + 1:]):
            can_be_connected = to_atom.GetFormalCharge() == 0 and from_atom.GetFormalCharge() == 0
            if can_be_connected:
                atom_pairs_to_connect.append((from_atom_id, to_atom_id))
    return atom_pairs_to_connect


def delete_bond(mol_graph: MolGraph,
                requirements: GraphRequirements,
                graph_gen_params: GraphGenerationParams = None,
                parameters: AlgorithmParameters = None):
    mol_graph = deepcopy(mol_graph)
    atom_pairs_to_disconnect = get_atom_pairs_to_disconnect(mol_graph)
    if atom_pairs_to_disconnect:
        pair_to_disonnect = choice(atom_pairs_to_disconnect)
        mol_graph.delete_bond(*pair_to_disonnect)
    return mol_graph


def get_atom_pairs_to_disconnect(mol_graph: MolGraph) -> Sequence[Tuple[int, int]]:
    molecule = mol_graph.get_rw_molecule()
    atoms = np.array(molecule.GetAtoms())
    atom_pairs_to_connect = []
    for from_atom_id, from_atom in enumerate(atoms):
        for to_atom_id, to_atom in enumerate(atoms[from_atom_id + 1:]):
            can_be_disconnected = to_atom.GetFormalCharge() == 0 and from_atom.GetFormalCharge() == 0
            if can_be_disconnected:
                atom_pairs_to_connect.append((from_atom_id, to_atom_id))
    raise NotImplementedError


if __name__ == '__main__':
    graph = deepcopy(MolGraph.from_smiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"))
    graph.show()
    new_graph = replace_bond(graph, MolGraphRequirements())
    new_graph.show()
