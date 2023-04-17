import os
import sys

from rdkit import RDConfig
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed
from rdkit.Chem.rdchem import RWMol

from examples.molecule_search.mol_graph import MolGraph

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def qed_score(mol_graph: MolGraph):
    score = qed(mol_graph.rw_molecule)
    return -score


def sa_score(mol_graph: MolGraph):
    score = sascorer.calculateScore(mol_graph.rw_molecule)
    return score


def penalised_logp(mol_graph: MolGraph):
    def largest_ring_size(rw_molecule: RWMol):
        largest_cycle_len = 0
        cycle_list = rw_molecule.GetRingInfo().AtomRings()
        if cycle_list:
            largest_cycle_len = max(list(map(len, cycle_list)))
        print(largest_cycle_len)
        return largest_cycle_len

    molecule = mol_graph.rw_molecule
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    score = log_p - sas_score - cycle_score
    return -score


def normalized_sa_score(mol_graph: MolGraph):
    score = sa_score(mol_graph)
    normalized_score = 1 - (score - 1) / 9
    return -normalized_score


def cl_score(mol_graph: MolGraph):
    # https://github.com/reymond-group/GDBChEMBL
    pass


if __name__ == '__main__':
    graph = MolGraph.from_smiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    print('QED', qed_score(graph))
    print('SA score', sa_score(graph))
    print('penalized LogP', penalised_logp(graph))
    print('normalized SA score', normalized_sa_score(graph))
