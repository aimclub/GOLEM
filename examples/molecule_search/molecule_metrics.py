import os
import pickle
import sys

from rdkit import RDConfig, Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem.rdchem import RWMol

from examples.molecule_search.mol_graph import MolGraph
from golem.core.paths import project_root

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def qed_score(mol_graph: MolGraph):
    molecule = mol_graph.get_rw_molecule(aromatic=True)
    score = qed(molecule)
    return -score


def sa_score(mol_graph: MolGraph):
    molecule = mol_graph.get_rw_molecule(aromatic=True)
    score = sascorer.calculateScore(molecule)
    return score


def penalised_logp(mol_graph: MolGraph):
    def largest_ring_size(rw_molecule: RWMol):
        largest_cycle_len = 0
        cycle_list = rw_molecule.GetRingInfo().AtomRings()
        if cycle_list:
            largest_cycle_len = max(list(map(len, cycle_list)))
        return largest_cycle_len

    molecule = mol_graph.get_rw_molecule(aromatic=True)
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


def cl_score(mol_graph: MolGraph, weighted: bool = True, radius: int = 3, rooted: bool = True):
    """Original code: https://github.com/reymond-group/GDBChEMBL """
    molecule = mol_graph.get_rw_molecule(aromatic=True)
    db_shingles = load_shingles(rooted)
    qry_shingles = _extract_shingles(molecule, radius, rooted)

    # calculate shingle count averaged score
    avg_score = 0
    if qry_shingles:
        sum_scores = 0
        # using log10 of shingle frequency
        if weighted:
            for shingle in qry_shingles:
                sum_scores += db_shingles.get(shingle, 0)
        # working binary (i.e. if present -> count++ )
        else:
            for shingle in qry_shingles:
                if shingle in db_shingles:
                    sum_scores += 1
        avg_score = sum_scores / len(qry_shingles)

    return -avg_score


def load_shingles(rooted: bool = True):
    if rooted:
        with open(os.path.join(project_root(),
                               "examples/molecule_search/data/shingles",
                               "chembl_24_1_shingle_scores_log10_rooted_nchir_min_freq_100.pkl"), "rb") as pyc:
            db_shingles = pickle.load(pyc)
    else:
        with open(os.path.join(project_root(),
                               "examples/molecule_search/data/shingles",
                               "chembl_24_1_shingle_scores_log10_nrooted_nchir.pkl"), "rb") as pyc:
            db_shingles = pickle.load(pyc)
    return db_shingles


def _extract_shingles(molecule: RWMol, radius: int = 3, rooted: bool = True):
    qry_shingles = set()

    radius_constr = radius + 1

    for atm_idx in range(molecule.GetNumAtoms()):
        for N in range(1, radius_constr):
            bonds = AllChem.FindAtomEnvironmentOfRadiusN(molecule, N, atm_idx)

            if not bonds:
                break

            atoms = set()
            for bond_id in bonds:
                bond = molecule.GetBondWithIdx(bond_id)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())

            if rooted:
                new_shingle = Chem.MolFragmentToSmiles(molecule,
                                                       atomsToUse=list(atoms),
                                                       bondsToUse=bonds,
                                                       isomericSmiles=False,
                                                       rootedAtAtom=atm_idx)
            else:
                new_shingle = Chem.MolFragmentToSmiles(molecule,
                                                       atomsToUse=list(atoms),
                                                       bondsToUse=bonds,
                                                       isomericSmiles=False,
                                                       rootedAtAtom=-1)

            qry_shingles.add(new_shingle)

    return qry_shingles
