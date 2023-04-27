from copy import deepcopy
from random import choice
from typing import Optional

import networkx as nx

from examples.molecule_search.mol_advisor import get_free_electrons_num
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.mol_graph_parameters import MolGraphRequirements
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams, AlgorithmParameters


def add_atom(mol_graph: MolGraph,
             requirements: MolGraphRequirements,
             graph_gen_params: GraphGenerationParams,
             parameters: Optional[AlgorithmParameters] = None):
    atoms_to_connect = graph_gen_params.advisor.propose_connection_point(mol_graph)
    if atoms_to_connect and mol_graph.heavy_atoms_number < requirements.max_heavy_atoms:
        connect_to_atom_id = int(choice(atoms_to_connect))
        connect_to_atom = mol_graph.get_rw_molecule().GetAtomWithIdx(connect_to_atom_id)

        atom_types_to_add = graph_gen_params.advisor.propose_parent(connect_to_atom, requirements.available_atom_types)

        if atom_types_to_add:
            atom_type_to_add = choice(atom_types_to_add)
            mol_graph.add_atom(atom_type_to_add)

            new_atom_id = mol_graph.heavy_atoms_number - 1
            mol_graph.set_bond(connect_to_atom_id, new_atom_id)
            return mol_graph
    else:
        return mol_graph


def delete_atom(mol_graph: MolGraph,
                requirements: GraphRequirements,
                graph_gen_params: GraphGenerationParams,
                parameters: Optional[AlgorithmParameters] = None):
    atoms_to_delete = graph_gen_params.advisor.propose_atom_removal(mol_graph)
    if atoms_to_delete:
        atom_to_delete_id = int(choice(atoms_to_delete))
        mol_graph.remove_atom(atom_to_delete_id)
    return mol_graph


def replace_atom(mol_graph: MolGraph,
                 requirements: MolGraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 parameters: Optional[AlgorithmParameters] = None):
    atom_to_replace = choice(mol_graph.get_rw_molecule().GetAtoms())
    possible_substitutions = graph_gen_params.advisor.propose_change(atom_to_replace, requirements.available_atom_types)
    if possible_substitutions:
        new_atom_type = choice(possible_substitutions)
        mol_graph.replace_atom(atom_to_replace.GetIdx(), new_atom_type)
    return mol_graph


def replace_bond(mol_graph: MolGraph,
                 requirements: MolGraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 parameters: Optional[AlgorithmParameters] = None):
    molecule = mol_graph.get_rw_molecule()
    atom_pairs_to_connect = graph_gen_params.advisor.propose_connection(mol_graph)

    if atom_pairs_to_connect:
        pair_to_connect = choice(atom_pairs_to_connect)
        from_atom = molecule.GetAtomWithIdx(pair_to_connect[0])
        to_atom = molecule.GetAtomWithIdx(pair_to_connect[1])

        pair_free_electrons = min(get_free_electrons_num(from_atom), get_free_electrons_num(to_atom))

        current_bond = molecule.GetBondBetweenAtoms(*pair_to_connect)
        current_bond_degree = current_bond.GetBondTypeAsDouble() if current_bond else 0

        max_bond_degree = current_bond_degree + pair_free_electrons
        possible_bonds = [bond for bond in requirements.bond_types if int(bond) <= max_bond_degree]
        if possible_bonds:
            bond_to_replace = choice(possible_bonds)
            mol_graph.set_bond(bond_type=bond_to_replace, *pair_to_connect)

    return mol_graph


def delete_bond(mol_graph: MolGraph,
                requirements: GraphRequirements,
                graph_gen_params: GraphGenerationParams,
                parameters: Optional[AlgorithmParameters] = None):
    atom_pairs_to_disconnect = graph_gen_params.advisor.propose_disconnection(mol_graph)
    if atom_pairs_to_disconnect:
        pair_to_disconnect = choice(atom_pairs_to_disconnect)
        mol_graph.remove_bond(*pair_to_disconnect)
    return mol_graph


def cut_atom(mol_graph: MolGraph,
             requirements: GraphRequirements,
             graph_gen_params: GraphGenerationParams,
             parameters: Optional[AlgorithmParameters] = None):
    atoms_to_cut = graph_gen_params.advisor.propose_cut(mol_graph)
    molecule = mol_graph.get_rw_molecule()
    if atoms_to_cut:
        atom_to_cut = choice(atoms_to_cut)
        neighbors_id = [neighbor.GetIdx() for neighbor in molecule.GetAtomWithIdx(atom_to_cut).GetNeighbors()]
        mol_graph.remove_bond(neighbors_id[0], atom_to_cut, update_representation=False)
        mol_graph.remove_bond(neighbors_id[1], atom_to_cut, update_representation=False)
        mol_graph.set_bond(*neighbors_id, update_representation=False)
        mol_graph.remove_atom(atom_to_cut)
    return mol_graph


def insert_carbon(mol_graph: MolGraph,
                  requirements: MolGraphRequirements,
                  graph_gen_params: GraphGenerationParams,
                  parameters: Optional[AlgorithmParameters] = None):
    if mol_graph.heavy_atoms_number < requirements.max_heavy_atoms:
        bonds_to_split = graph_gen_params.advisor.propose_bond_to_split(mol_graph)
        if bonds_to_split:
            bond_to_split = choice(bonds_to_split)
            mol_graph.remove_bond(*bond_to_split, update_representation=False)
            mol_graph.add_atom('C')
            carbon_id = mol_graph.heavy_atoms_number - 1
            mol_graph.set_bond(bond_to_split[0], carbon_id, update_representation=False)
            mol_graph.set_bond(bond_to_split[1], carbon_id)
    return mol_graph


def remove_group(mol_graph: MolGraph,
                 requirements: MolGraphRequirements,
                 graph_gen_params: GraphGenerationParams,
                 parameters: Optional[AlgorithmParameters] = None):
    nx_graph = mol_graph.get_nx_graph()
    bridges_to_remove = graph_gen_params.advisor.propose_group(mol_graph)
    if bridges_to_remove:
        bridge = choice(bridges_to_remove)
        disconnected_graph = deepcopy(nx_graph)
        disconnected_graph.remove_edge(*bridge)
        group = list(nx.node_connected_component(disconnected_graph, bridge[1]))
        for atom_id in sorted(group, reverse=True):
            mol_graph.remove_atom(atom_id, update_representation=False)
        mol_graph.update_representation()
    return mol_graph


def move_group(mol_graph: MolGraph,
               requirements: MolGraphRequirements,
               graph_gen_params: GraphGenerationParams,
               parameters: Optional[AlgorithmParameters] = None):
    molecule = mol_graph.get_rw_molecule()
    bridges_to_move = graph_gen_params.advisor.propose_group(mol_graph)
    if bridges_to_move:
        bridge = choice(bridges_to_move)
        current_bond = molecule.GetBondBetweenAtoms(*bridge)

        disconnected_graph = deepcopy(mol_graph.get_nx_graph())
        disconnected_graph.remove_edge(*bridge)
        group = set(nx.node_connected_component(disconnected_graph, bridge[1]))

        atoms_to_reconnect_to = graph_gen_params.advisor.propose_connection_point(mol_graph, current_bond)
        atoms_to_reconnect_to = [atom_idx for atom_idx in atoms_to_reconnect_to if atom_idx not in group.union(bridge)]
        if atoms_to_reconnect_to:
            new_endpoint = int(choice(atoms_to_reconnect_to))
            mol_graph.remove_bond(*bridge, update_representation=False)
            mol_graph.set_bond(new_endpoint, bridge[1], bond_type=current_bond.GetBondType())
    return mol_graph
