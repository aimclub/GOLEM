from typing import List, Sequence, Tuple, Any

import networkx as nx
import numpy as np
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdchem import Atom

from examples.molecule_search.constants import SULFUR_DEFAULT_VALENCE
from examples.molecule_search.mol_graph import MolGraph
from golem.core.optimisers.advisor import DefaultChangeAdvisor


class MolChangeAdvisor(DefaultChangeAdvisor):
    def propose_parent(self, node: Atom, possible_operations: List[str]):
        atom_types = list(set(possible_operations) - set(node.GetSymbol()))
        return atom_types

    def propose_change(self, node: Atom, possible_operations: List[Any]):
        atom_types = list(set(possible_operations) - set(node.GetSymbol()))
        atom_types_to_replace = [atom_type for atom_type in atom_types
                                 if get_default_valence(atom_type) > node.GetExplicitValence()]
        return atom_types_to_replace

    @staticmethod
    def propose_connection_point(mol_graph: MolGraph) -> Sequence[int]:
        molecule = mol_graph.get_rw_molecule()
        atoms = molecule.GetAtoms()
        free_electrons_vector = np.array([get_free_electrons_num(atom) for atom in atoms])
        formal_charge_vector = np.array([atom.GetFormalCharge() for atom in atoms])
        atom_ids = np.arange(mol_graph.heavy_atoms_number)
        return list(atom_ids[(formal_charge_vector == 0) & (free_electrons_vector > 0)])

    @staticmethod
    def propose_atom_removal(mol_graph: MolGraph) -> Sequence[int]:
        nx_graph = mol_graph.get_nx_graph()
        art_points_ids = nx.articulation_points(nx_graph)
        atom_ids = np.arange(mol_graph.heavy_atoms_number)
        return list(set(atom_ids) - set(art_points_ids))

    @staticmethod
    def propose_connection(mol_graph: MolGraph) -> Sequence[Tuple[int, int]]:
        molecule = mol_graph.get_rw_molecule()
        atoms = np.array(molecule.GetAtoms())
        atom_pairs_to_connect = []
        for from_atom in atoms:
            for to_atom in atoms[from_atom.GetIdx() + 1:]:
                can_be_connected = to_atom.GetFormalCharge() == 0 and from_atom.GetFormalCharge() == 0
                if can_be_connected:
                    atom_pairs_to_connect.append((from_atom.GetIdx(), to_atom.GetIdx()))
        return atom_pairs_to_connect

    @staticmethod
    def propose_disconnection(mol_graph: MolGraph) -> Sequence[Tuple[int, int]]:
        molecule = mol_graph.get_rw_molecule()
        atoms = np.array(molecule.GetAtoms())
        bridges = set(nx.bridges(mol_graph.get_nx_graph()))
        atom_pairs_to_disconnect = []
        for from_atom in atoms:
            for to_atom in atoms[from_atom.GetIdx() + 1:]:
                if molecule.GetBondBetweenAtoms(from_atom.GetIdx(), to_atom.GetIdx()):
                    can_be_disconnected = to_atom.GetFormalCharge() == 0 and from_atom.GetFormalCharge() == 0
                    if can_be_disconnected:
                        atom_pairs_to_disconnect.append((from_atom.GetIdx(), to_atom.GetIdx()))
        atom_pairs_to_disconnect = set(atom_pairs_to_disconnect) - bridges
        return list(atom_pairs_to_disconnect)

    @staticmethod
    def propose_cut(mol_graph: MolGraph) -> Sequence[int]:
        atoms_to_cut = [atom.GetIdx() for atom in mol_graph.get_rw_molecule().GetAtoms()
                        if len(atom.GetNeighbors()) == 2]
        return atoms_to_cut


def get_default_valence(atom_type: str) -> int:
    if atom_type == "S":
        default_valence = SULFUR_DEFAULT_VALENCE
    else:
        default_valence = GetPeriodicTable().GetDefaultValence(atom_type)
    return default_valence


def get_free_electrons_num(atom: Atom) -> int:
    atom.UpdatePropertyCache()
    atom_type = atom.GetSymbol()
    default_valence = get_default_valence(atom_type)
    explicit_valence = atom.GetExplicitValence()
    free_electrons = default_valence - explicit_valence
    return free_electrons
