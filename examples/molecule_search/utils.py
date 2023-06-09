from copy import deepcopy

import networkx as nx
from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdchem import Atom, RWMol
from typing import Tuple, Set

from examples.molecule_search.constants import SULFUR_DEFAULT_VALENCE
from examples.molecule_search.mol_graph import MolGraph


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


def get_functional_group(mol_graph: MolGraph, bridge: Tuple[int, int]) -> Set[int]:
    disconnected_graph = mol_graph.get_nx_graph()
    disconnected_graph.remove_edge(*bridge)
    group = nx.node_connected_component(disconnected_graph, bridge[1])
    return group


def largest_ring_size(rw_molecule: RWMol) -> int:
    largest_cycle_len = 0
    cycle_list = rw_molecule.GetRingInfo().AtomRings()
    if cycle_list:
        largest_cycle_len = max(map(len, cycle_list))
    return largest_cycle_len
