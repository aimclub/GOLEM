from typing import List

from rdkit.Chem.rdchem import BondType

from golem.core.optimisers.optimization_parameters import GraphRequirements


class MolGraphRequirements(GraphRequirements):
    max_heavy_atoms: int = 50
    available_operations: List[str] = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
    bond_types: List[BondType] = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
