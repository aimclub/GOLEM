import os
import pickle
import sys
from typing import Dict

from rdkit import RDConfig, Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.QED import qed
from rdkit.Chem.rdchem import RWMol

from examples.molecule_search.constants import ZINC_LOGP_MEAN, ZINC_LOGP_STD, ZINC_SA_MEAN, ZINC_SA_STD, \
    ZINC_CYCLE_MEAN, ZINC_CYCLE_STD, MIN_LONG_CYCLE_SIZE
from examples.molecule_search.mol_graph import MolGraph
from examples.molecule_search.utils import largest_ring_size, download_from_github
from golem.core.paths import project_root

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def qed_score(mol_graph: MolGraph):
    """Quantitative estimation of drug-likeness. QED measure reflects the underlying distribution of molecular
    properties including molecular weight, logP, topological polar surface area, number of hydrogen bond donors and
    acceptors, the number of aromatic rings and rotatable bonds, and the presence of unwanted chemical
    functionalities.

    It is a numerical value ranging from 0 to 1, with a higher score indicating a greater likelihood
    of the molecule having favorable drug-like properties.
    """

    molecule = mol_graph.get_rw_molecule(aromatic=True)
    score = qed(molecule)
    return -score


def sa_score(mol_graph: MolGraph) -> float:
    """Synthetic Accessibility score is a metric used to evaluate the ease of synthesizing a molecule.
    The SA score takes into account a variety of factors such as the number of synthetic steps,
    the availability of starting materials, and the feasibility of the reaction conditions required for each step.

    It is ranged between 1 and 10, 1 is the best possible score.
    """

    molecule = mol_graph.get_rw_molecule(aromatic=True)
    score = sascorer.calculateScore(molecule)
    return score


def penalised_logp(mol_graph: MolGraph) -> float:
    """ LogP penalized by SA and length of long cycles, as described in (Kusner et al. 2017).
    LogP (Octanol-Water Partition Coefficient) is a measure of the lipophilicity,
    or ability to dissolve in lipids, of a molecule. It is commonly used to predict the bioavailability
    and pharmacokinetic properties of drugs.

    This version is penalized by SA score and the length of the largest cycle.
    """

    molecule = mol_graph.get_rw_molecule(aromatic=True)
    log_p = Descriptors.MolLogP(molecule)
    sa_score = sascorer.calculateScore(molecule)
    largest_cycle_size = largest_ring_size(molecule)
    cycle_score = max(largest_cycle_size - MIN_LONG_CYCLE_SIZE, 0)
    score = log_p - sa_score - cycle_score
    return -score


def normalized_logp(mol_graph: MolGraph) -> float:
    """Normalized LogP based on the statistics of 250k_rndm_zinc_drugs_clean.smi dataset.

    Original code:
    https://github.com/bowenliu16/rl_graph_generation/blob/master/gym-molecule/gym_molecule/envs/molecule.py
    """
    molecule = mol_graph.get_rw_molecule(aromatic=True)

    logp = Descriptors.MolLogP(molecule)
    sa = -sascorer.calculateScore(molecule)

    largest_ring = largest_ring_size(molecule)
    cycle_score = max(largest_ring - 6, 0)

    norm_logp = (logp - ZINC_LOGP_MEAN) / ZINC_LOGP_STD
    normalized_sa = (sa - ZINC_SA_MEAN) / ZINC_SA_STD
    normalized_cycle = (cycle_score - ZINC_CYCLE_MEAN) / ZINC_CYCLE_STD

    score = norm_logp + normalized_sa + normalized_cycle

    return score


def normalized_sa_score(mol_graph: MolGraph) -> float:
    """SA score normalized to be ranged from 0 to 1, where 1 is preferable."""

    score = sa_score(mol_graph)
    normalized_score = 1 - (score - 1) / 9
    return -normalized_score


class CLScorer:
    """
    ChEMBL-likeness score (CLscore) is defined by considering which substructures in a molecule
    also occur in molecules from the public database ChEMBL, using a subset of molecules with
    reported high confidence datapoint of activity on single protein targets.

    Original code: https://github.com/reymond-group/GDBChEMBL

    Args:
        weighted: If True calculate score by summing up log10 of frequency of occurrence inside the reference database.
            When set to False, score will add 1 for every shingle inside the reference database,
            irrespective of how often it occurs. It is recommended to use the actual CLscore with weighted shingles.
        radius: Maximum radius of circular substructures around the rooting atom.
            Note that when using the ChEMBL shingle library, maximum radius will be 3 (default).
            For larger radii, the reference database would have to be read out accordingly.
        rooted: Use rooted shingles. This means reference shingles are canonical but always starting at the central
            atom of a circular substructure. False means shingles in the database are canonicalized but not rooted.
            It is recommended to use rooted shingles.
    """
    ROOTED_FILE_PATH = "examples/molecule_search/data/shingles/" \
                       "chembl_24_1_shingle_scores_log10_rooted_nchir_min_freq_100.pkl"
    NOT_ROOTED_FILE_PATH = "examples/molecule_search/data/shingles/chembl_24_1_shingle_scores_log10_nrooted_nchir.pkl"

    ROOTED_GITHUB_URL = "https://github.com/aimclub/GOLEM/raw/molecule-search/examples/molecule_search/data/shingles/" \
                        "chembl_24_1_shingle_scores_log10_rooted_nchir_min_freq_100.pkl"
    NOT_ROOTED_GITHUB_URL = "https://github.com/aimclub/GOLEM/raw/molecule-search/examples/molecule_search/data/" \
                            "shingles/chembl_24_1_shingle_scores_log10_nrooted_nchir.pkl"

    def __init__(self, weighted: bool = True, radius: int = 3, rooted: bool = True):
        self.rooted = rooted
        self.weighted = weighted
        self.radius = radius

        file_path = CLScorer.ROOTED_FILE_PATH if self.rooted else CLScorer.NOT_ROOTED_FILE_PATH
        self.file_path = os.path.join(project_root(), file_path)
        self.github_url = CLScorer.ROOTED_GITHUB_URL if self.rooted else CLScorer.NOT_ROOTED_GITHUB_URL
        self.db_shingles = self.load_shingles()

    def __call__(self, mol_graph: MolGraph) -> float:
        """
        Args:
            mol_graph: MolGraph to evaluate
        """
        molecule = mol_graph.get_rw_molecule(aromatic=True)
        qry_shingles = self.extract_shingles(molecule)

        # calculate shingle count averaged score
        avg_score = 0
        if qry_shingles:
            sum_scores = 0
            # using log10 of shingle frequency
            if self.weighted:
                for shingle in qry_shingles:
                    sum_scores += self.db_shingles.get(shingle, 0)
            # working binary (i.e. if present -> count++ )
            else:
                for shingle in qry_shingles:
                    if shingle in self.db_shingles:
                        sum_scores += 1
            avg_score = sum_scores / len(qry_shingles)

        return -avg_score

    def load_shingles(self) -> Dict:
        download_from_github(self.file_path, self.github_url)

        with open(self.file_path, "rb") as pyc:
            db_shingles = pickle.load(pyc)
        return db_shingles

    def extract_shingles(self, molecule: RWMol) -> set:
        qry_shingles = set()

        radius_constr = self.radius + 1

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

                if self.rooted:
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
