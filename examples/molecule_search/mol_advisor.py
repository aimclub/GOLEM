from random import choice

from rdkit.Chem.rdchem import Atom
from typing import List


class MolChangeAdvisor:
    @staticmethod
    def propose_atom_type(atom: Atom, available_types: List[str]):
        atom_types = list(set(available_types) - set(atom.GetSymbol()))
        return choice(atom_types)
