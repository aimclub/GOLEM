import io

import networkx as nx
from PIL import Image
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, SanitizeMol
from rdkit.Chem.Draw import rdMolDraw2D


class MolGraph:
    def __init__(self, rw_molecule: Chem.RWMol):
        self.rw_molecule = rw_molecule

    @staticmethod
    def from_smiles(smiles: str):
        rw_molecule = MolFromSmiles(smiles)
        return MolGraph(rw_molecule)

    @staticmethod
    def from_nx_graph(graph: nx.Graph):
        mol = Chem.RWMol()
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
    def nx_graph(self) -> nx.Graph:
        graph = nx.Graph()

        for atom in self.rw_molecule.GetAtoms():
            graph.add_node(atom.GetIdx(),
                           atomic_num=atom.GetAtomicNum(),
                           formal_charge=atom.GetFormalCharge(),
                           chiral_tag=atom.GetChiralTag(),
                           hybridization=atom.GetHybridization(),
                           num_explicit_hs=atom.GetNumExplicitHs(),
                           is_aromatic=atom.GetIsAromatic())
        for bond in self.rw_molecule.GetBonds():
            graph.add_edge(bond.GetBeginAtomIdx(),
                           bond.GetEndAtomIdx(),
                           bond_type=bond.GetBondType())
        return graph

    @property
    def smiles(self) -> str:
        return MolToSmiles(self.rw_molecule)

    def show(self):
        drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(self.rw_molecule)
        drawer.FinishDrawing()
        bytes_images = drawer.GetDrawingText()
        image = Image.open(io.BytesIO(bytes_images))
        image.show()