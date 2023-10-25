import io

import networkx as nx
from PIL import Image
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, SanitizeMol, Kekulize, MolToInchi
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Atom, BondType, RWMol, GetPeriodicTable, ChiralType, HybridizationType


class MolGraph:
    def __init__(self, rw_molecule: RWMol):
        self._rw_molecule = rw_molecule
        self.update_representation()

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
            a.SetChiralTag(ChiralType(chiral_tags[node]))
            a.SetFormalCharge(formal_charges[node])
            a.SetIsAromatic(node_is_aromatics[node])
            a.SetHybridization(HybridizationType(node_hybridizations[node]))
            a.SetNumExplicitHs(num_explicit_hss[node])
            idx = mol.AddAtom(a)
            node_to_idx[node] = idx

        bond_types = nx.get_edge_attributes(graph, 'bond_type')
        for edge in graph.edges():
            first, second = edge
            ifirst = node_to_idx[first]
            isecond = node_to_idx[second]
            bond_type = BondType(bond_types[first, second])
            mol.AddBond(ifirst, isecond, bond_type)

        SanitizeMol(mol)
        return MolGraph(mol)

    @property
    def heavy_atoms_number(self) -> int:
        return self._rw_molecule.GetNumAtoms()

    def get_nx_graph(self) -> nx.Graph:
        """Original code: https://github.com/maxhodak/keras-molecules"""
        graph = nx.Graph()

        for atom in self._rw_molecule.GetAtoms():
            graph.add_node(atom.GetIdx(),
                           # save indices in node data to restore edges parameters after adapt-restore process
                           nxid=str(atom.GetIdx()),
                           name=atom.GetSymbol(),
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
        return MolToSmiles(self.get_rw_molecule(aromatic))

    def get_rw_molecule(self, aromatic: bool = False) -> RWMol:
        if aromatic:
            return MolFromSmiles(self.get_smiles())
        else:
            return self._rw_molecule

    def update_representation(self):
        SanitizeMol(self._rw_molecule)
        Kekulize(self._rw_molecule)

        # Setting all bonds and atoms to non aromatics
        for atom_from in self._rw_molecule.GetAtoms():
            atom_from.SetIsAromatic(False)
            for atom_to in self._rw_molecule.GetAtoms():
                bond = self._rw_molecule.GetBondBetweenAtoms(atom_from.GetIdx(), atom_to.GetIdx())
                if bond is not None:
                    bond.SetIsAromatic(False)

        # Updating the property cache of atoms
        for atom in self._rw_molecule.GetAtoms():
            atom.UpdatePropertyCache()

        self._rw_molecule.UpdatePropertyCache()

    def add_atom(self, atom_type: str):
        atom = Atom(atom_type)
        self._rw_molecule.AddAtom(atom)

    def set_bond(self, from_atom: int, to_atom: int, bond_type: BondType = BondType.SINGLE, update_representation=True):
        current_bond = self._rw_molecule.GetBondBetweenAtoms(from_atom, to_atom)
        if current_bond is None:
            self._rw_molecule.AddBond(from_atom, to_atom, bond_type)
        else:
            current_bond.SetBondType(bond_type)
        if update_representation:
            self.update_representation()

    def remove_bond(self, from_atom: int, to_atom: int, update_representation=True):
        self._rw_molecule.RemoveBond(from_atom, to_atom)
        if update_representation:
            self.update_representation()

    def remove_atom(self, atom_id: int, update_representation=True):
        self._rw_molecule.RemoveAtom(atom_id)
        if update_representation:
            self.update_representation()

    def replace_atom(self, atom_to_replace: int, atom_type: str):
        new_atomic_number = GetPeriodicTable().GetAtomicNumber(atom_type)

        self._rw_molecule.GetAtomWithIdx(atom_to_replace).SetAtomicNum(new_atomic_number)

        self._rw_molecule.GetAtomWithIdx(atom_to_replace).SetFormalCharge(0)

        self.update_representation()

    def show(self):
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(self._rw_molecule)
        drawer.FinishDrawing()
        bytes_images = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(bytes_images))
        image.show()

    def __eq__(self, other_mol: 'MolGraph'):
        return MolToInchi(self.get_rw_molecule(aromatic=True)) == MolToInchi(other_mol.get_rw_molecule(aromatic=True))

    def __hash__(self):
        return hash(MolToInchi(self.get_rw_molecule(aromatic=True)))

    def __str__(self):
        return self.get_smiles()
