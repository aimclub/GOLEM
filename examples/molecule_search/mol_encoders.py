import os
from typing import Any, List, Optional

import numpy as np
import torch
from gensim.models import word2vec, Word2Vec
from mol2vec.features import mol2alt_sentence, MolSentence
from rdkit.Chem import AllChem, RDKFingerprint, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_transformer.transformer import create_masks, Transformer, EXTRA_CHARS, ALPHABET_SIZE
from examples.molecule_search.utils import download_from_github
from golem.core.log import default_log
from golem.core.paths import project_root


def adapter_func_to_molgraph(func):
    """ Decorator function to adapt observation to MolGraphs graphs. """
    def wrapper(obs):
        mol_graph = MolAdapter().restore(obs)
        embedding = func(mol_graph)
        return embedding
    return wrapper


def adapter_method_to_molgraph(func):
    """ Decorator function to adapt observation to MolGraphs graphs. """
    def wrapper(obj, obs):
        mol_graph = MolAdapter().restore(obs)
        embedding = func(obj, mol_graph)
        return embedding
    return wrapper


@adapter_func_to_molgraph
def ECFP(obs: Any):
    """ Extended-Connectivity Fingerprint """
    molecule = obs.get_rw_molecule()
    feature_list = AllChem.GetMorganFingerprintAsBitVect(molecule,
                                                         radius=2,
                                                         nBits=2**10,
                                                         useFeatures=False,
                                                         useChirality=False)
    return np.array(feature_list)


@adapter_func_to_molgraph
def RDKF(obs: Any):
    """ RDK Fingerprint """
    molecule = obs.get_rw_molecule()
    fingerprint_rdk = RDKFingerprint(molecule)
    return np.array(fingerprint_rdk)


@adapter_func_to_molgraph
def atom_pair(obs: Any):
    """ Atom pair fingerprint """
    molecule = obs.get_rw_molecule()
    fingerprint = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=1024).GetFingerprint(molecule)
    return np.array(fingerprint)


@adapter_func_to_molgraph
def topological_torsion(obs: Any):
    """ Topological Torsion fingerprint """
    molecule = obs.get_rw_molecule()
    fingerprint = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=1024).GetFingerprint(molecule)
    return np.array(fingerprint)


@adapter_func_to_molgraph
def mol_descriptors(obs: Any):
    molecule = obs.get_rw_molecule()
    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
                          'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
                          'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
                          'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
                          'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                          'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
                          'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge',
                          'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount',
                          'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
                          'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
                          'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
                          'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
                          'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
                          'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                          'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10',
                          'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8',
                          'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
                          'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8',
                          'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
                          'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
                          'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH',
                          'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine',
                          'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                          'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                          'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide',
                          'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
                          'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
                          'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                          'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                          'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                          'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                          'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
                          'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd',
                          'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
                          'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
    mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)
    list_of_descriptor_vals = list(mol_descriptor_calculator.CalcDescriptors(molecule))
    return list_of_descriptor_vals


class Mol2Vec:

    PRETRAINED_WORD2VEC = 'examples/molecule_search/data/pretrained_models/model_300dim.pkl'
    GITHUB_URL = 'https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl'

    def __init__(self):
        self.file_path = os.path.join(project_root(), Mol2Vec.PRETRAINED_WORD2VEC)
        download_from_github(self.file_path,
                             Mol2Vec.GITHUB_URL,
                             message="Downloading pretrained model for molecules encoding...")

        self.model = word2vec.Word2Vec.load(self.file_path)

    @adapter_method_to_molgraph
    def __call__(self, obs: Any):
        molecule = obs.get_rw_molecule()
        sentence = MolSentence(mol2alt_sentence(molecule, radius=1))
        embedding = self.sentences2vec([sentence], self.model, unseen='UNK')[0]
        return np.array(embedding).astype(float)

    @staticmethod
    def sentences2vec(sentences: List[MolSentence], model: Word2Vec, unseen: Optional[str] = None) -> np.array:
        """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
        sum of vectors for individual words.

        Parameters
        ----------
        sentences : list, array
            List with sentences
        model : word2vec.Word2Vec
            Gensim word2vec model
        unseen : None, str
            Keyword for unseen words. If None, those words are skipped.
            https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

        Returns
        -------
        np.array
        """

        keys = set(model.wv.key_to_index)
        vec = []

        if unseen:
            unseen_vec = model.wv.get_vector(unseen)

        for sentence in sentences:
            if unseen:
                vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys
                                else unseen_vec for y in sentence]))
            else:
                vec.append(sum([model.wv.get_vector(y) for y in sentence
                                if y in set(sentence) & keys]))
        return np.array(vec)


class MoleculeTransformer:
    """ Based on https://github.com/mpcrlab/MolecularTransformerEmbeddings """

    PRETRAINED_TRANSFORMER = 'examples/molecule_search/data/pretrained_models/pretrained.ckpt'
    GITHUB_URL = 'https://github.com/mpcrlab/MolecularTransformerEmbeddings/releases/download/' \
                 'checkpoints/pretrained.ckpt'

    def __init__(self, embedding_size: int = 512, num_layers: int = 6, max_length: int = 256):
        self.log = default_log(self)

        self.file_path = os.path.join(project_root(), MoleculeTransformer.PRETRAINED_TRANSFORMER)
        download_from_github(self.file_path,
                             MoleculeTransformer.GITHUB_URL,
                             message="Downloading pretrained model for molecules encoding...")
        self.model = self._model_setup(embedding_size, num_layers)
        self.encoder = self.model.encoder.cpu()
        self.max_length = max_length

    def _model_setup(self, embedding_size: int, num_layers: int):
        model = Transformer(ALPHABET_SIZE, embedding_size, num_layers).eval()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(self.file_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        return model.module.cpu()

    @adapter_method_to_molgraph
    def __call__(self, obs: Any):
        smiles = obs.get_smiles()
        with torch.no_grad():
            encoded = self.encode_smiles(smiles)
            mask = create_masks(encoded)
            embedding = self.encoder(encoded, mask)[0].numpy()
            embedding = embedding.mean(axis=0)
        return embedding

    @staticmethod
    def encode_char(c):
        return ord(c) - 32

    def encode_smiles(self, string: str, start_char=EXTRA_CHARS['seq_start']):
        return torch.tensor([ord(start_char)] +
                            [self.encode_char(c) for c in string], dtype=torch.long)[:self.max_length].unsqueeze(0)
