from dataclasses import dataclass, field
from typing import List, Sequence

from rdkit.Chem.rdchem import BondType

from golem.core.optimisers.optimization_parameters import GraphRequirements


@dataclass
class MolGraphRequirements(GraphRequirements):
    max_heavy_atoms: int = 50
    available_atom_types: List[str] = field(default_factory=lambda: ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'])
    bond_types: Sequence[BondType] = field(default_factory=lambda: [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE])
