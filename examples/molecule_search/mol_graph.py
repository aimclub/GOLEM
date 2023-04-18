import io

import networkx as nx
from PIL import Image
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, SanitizeMol
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Atom, BondType, RWMol


class MolGraph:
    def __init__(self, rw_molecule: RWMol):
        self._rw_molecule = rw_molecule

    @staticmethod
    def from_smiles(smiles: str):
        molecule = MolFromSmiles(smiles)
        rw_molecule = RWMol(molecule)
        return MolGraph(rw_molecule)

    @staticmethod
    def from_nx_graph(graph: nx.Graph):
        """Original code: https://github.com/maxhodak/keras-molecules"""
        mol = RWMol()
        atomic_nums = nx.get_node_attributes(graph, 'atomic_num')
        chiral_tags = nx.get_node_attributes(graph, 'chiral_tag')
        formal_charges = nx.get_node_attributes(graph, 'formal_charge')
        node_is_aromatics = nx.get_node_attributes(graph, 'is_aromatic')
        node_hybridizations = nx.get_node_attributes(graph, 'hybridization')
        num_explicit_hss = nx.get_node_attributes(graph, 'num_explicit_hs')
        node_to_idx = {}
        for node in graph.nodes():
            a = Chem.Atom(atomic_nums[node])
            a.SetChiralTag(chiral_tags[node])
            a.SetFormalCharge(formal_charges[node])
            a.SetIsAromatic(node_is_aromatics[node])
            a.SetHybridization(node_hybridizations[node])
            a.SetNumExplicitHs(num_explicit_hss[node])
            idx = mol.AddAtom(a)
            node_to_idx[node] = idx

        bond_types = nx.get_edge_attributes(graph, 'bond_type')
        for edge in graph.edges():
            first, second = edge
            ifirst = node_to_idx[first]
            isecond = node_to_idx[second]
            bond_type = bond_types[first, second]
            mol.AddBond(ifirst, isecond, bond_type)

        SanitizeMol(mol)
        return mol

    @property
    def heavy_atoms_number(self) -> int:
        return self._rw_molecule.GetNumAtoms()

    def get_nx_graph(self) -> nx.Graph:
        """Original code: https://github.com/maxhodak/keras-molecules"""
        graph = nx.Graph()

        for atom in self._rw_molecule.GetAtoms():
            graph.add_node(atom.GetIdx(),
                           atomic_num=atom.GetAtomicNum(),
                           formal_charge=atom.GetFormalCharge(),
                           chiral_tag=atom.GetChiralTag(),
                           hybridization=atom.GetHybridization(),
                           num_explicit_hs=atom.GetNumExplicitHs(),
                           is_aromatic=atom.GetIsAromatic())
        for bond in self._rw_molecule.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(),
                           bond.GetEndAtomIdx(),
                           bond_type=bond.GetBondType())
        return graph

    def get_smiles(self, aromatic: bool = False) -> str:
        return MolToSmiles(self._rw_molecule)

    def get_rw_molecule(self, aromatic: bool = False) -> RWMol:
        return self._rw_molecule

    def add_atom(self, atom_type: str):
        atom = Atom(atom_type)
        atom.SetBoolProp("mutability", True)
        self._rw_molecule.AddAtom(atom)

    def add_bond(self, from_atom, to_atom):
        self._rw_molecule.AddBond(from_atom, to_atom, BondType.SINGLE)

    def show(self):
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(self._rw_molecule)
        drawer.FinishDrawing()
        bytes_images = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(bytes_images))
        image.show()
