from typing import List, Sequence, Tuple, Optional

import networkx as nx
import numpy as np
from rdkit.Chem.rdchem import Atom, Bond

from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.utils import get_default_valence, get_free_electrons_num
from golem.core.optimisers.advisor import DefaultChangeAdvisor


class MolChangeAdvisor(DefaultChangeAdvisor):
    def propose_parent(self, node: Atom, possible_operations: List[str]) -> List[str]:
        """
        Proposes types of atoms that can be connected to the current atom -
        any type other than the type of the current atom.
        """
        atom_types = list(set(possible_operations) - set(node.GetSymbol()))
        return atom_types

    def propose_change(self, node: Atom, possible_operations: List[str]) -> List[str]:
        """
        Proposes types of atoms to replace current atom type - any type other than the type of the current atom
        such that it has enough valence number (valence must be greater or equal to current atom explicit valence).
        """
        atom_types = list(set(possible_operations) - set(node.GetSymbol()))
        atom_types_to_replace = [atom_type for atom_type in atom_types
                                 if get_default_valence(atom_type) >= node.GetExplicitValence()]
        return atom_types_to_replace

    @staticmethod
    def propose_connection_point(mol_graph: MolGraph, bond: Optional[Bond] = None) -> Sequence[int]:
        """
        Proposes atoms new atom or functional group can connect to -
        atoms must have zero formal charge and enough free electrons.
        """
        molecule = mol_graph.get_rw_molecule()
        atoms = molecule.GetAtoms()
        free_electrons_vector = np.array([get_free_electrons_num(atom) for atom in atoms])
        formal_charge_vector = np.array([atom.GetFormalCharge() for atom in atoms])
        atom_ids = np.arange(mol_graph.heavy_atoms_number)

        bond_degree = bond.GetBondTypeAsDouble() if bond else 1  # BondType.SINGLE is used by default
        return list(atom_ids[(formal_charge_vector == 0) & (free_electrons_vector >= bond_degree)])

    @staticmethod
    def propose_atom_removal(mol_graph: MolGraph) -> Sequence[int]:
        """
        Proposes atoms that can be removed - any atom, which deletion will not increase the number of connected
        components of the molecule.
        """
        if mol_graph.heavy_atoms_number > 1:
            nx_graph = mol_graph.get_nx_graph()
            art_points_ids = nx.articulation_points(nx_graph)
            atom_ids = np.arange(mol_graph.heavy_atoms_number)
            return list(set(atom_ids) - set(art_points_ids))
        else:
            return []

    @staticmethod
    def propose_connection(mol_graph: MolGraph) -> Sequence[Tuple[int, int]]:
        """
        Proposes pairs of atoms that can be connected - both atoms must have zero formal charge.
        """
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
        """
        Proposes pairs of atoms that can be disconnected - both atoms must have zero formal charge and deletion
        of the bond must not increase the number of connected components of the molecule.
        """
        bridges = set(nx.bridges(mol_graph.get_nx_graph()))
        atom_pairs_to_disconnect = []
        for bond in mol_graph.get_rw_molecule().GetBonds():
            from_atom = bond.GetBeginAtom()
            to_atom = bond.GetEndAtom()
            can_be_disconnected = to_atom.GetFormalCharge() == 0 and from_atom.GetFormalCharge() == 0
            if can_be_disconnected:
                atom_pairs_to_disconnect.append((from_atom.GetIdx(), to_atom.GetIdx()))
        atom_pairs_to_disconnect = set(atom_pairs_to_disconnect) - bridges
        return list(atom_pairs_to_disconnect)

    @staticmethod
    def propose_cut(mol_graph: MolGraph) -> Sequence[int]:
        """
        Proposes atoms that can be cut - any atom that has only two neighbors, the neighbors are disconnected,
        and they have zero formal charge.
        """
        molecule = mol_graph.get_rw_molecule()
        atoms_to_cut = []
        for atom in molecule.GetAtoms():
            neighbors = atom.GetNeighbors()
            if len(neighbors) == 2:
                if (not molecule.GetBondBetweenAtoms(neighbors[0].GetIdx(), neighbors[1].GetIdx()) and
                        neighbors[0].GetFormalCharge() == 0 and
                        neighbors[1].GetFormalCharge() == 0):
                    atoms_to_cut.append(atom.GetIdx())
        return atoms_to_cut

    @staticmethod
    def propose_bond_to_split(mol_graph: MolGraph) -> Sequence[Tuple[int, int]]:
        """
        Proposes bonds which can be split by new atom - bond endpoints have no formal charge.
        """
        bonds_to_split = []
        for bond in mol_graph.get_rw_molecule().GetBonds():
            from_atom = bond.GetBeginAtom()
            to_atom = bond.GetEndAtom()
            if from_atom.GetFormalCharge() == 0 and to_atom.GetFormalCharge() == 0:
                bonds_to_split.append((from_atom.GetIdx(), to_atom.GetIdx()))
        return bonds_to_split

    @staticmethod
    def propose_group(mol_graph: MolGraph) -> List[Tuple[int, int]]:
        """
        Proposes functional groups that can be operated. Group is a connected subgraph that is connected
        to the rest of the graph by only one bond.

        Returns:
            List of indices pairs which are endpoints of bridges. This bridges connect functional group to the molecule.
            The second index of each pair belongs to the functional group.
        """
        molecule = mol_graph.get_rw_molecule()
        nx_graph = mol_graph.get_nx_graph()
        groups = []
        for bridge in nx.bridges(nx_graph):
            from_atom = molecule.GetAtomWithIdx(bridge[0])
            to_atom = molecule.GetAtomWithIdx(bridge[1])
            can_be_disconnected = to_atom.GetFormalCharge() == 0 and from_atom.GetFormalCharge() == 0
            if can_be_disconnected:
                groups.extend([bridge, bridge[::-1]])
        return groups
