import os
from typing import Any

import numpy as np
import requests
import torch
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence
from rdkit.Chem import AllChem, RDKFingerprint

from examples.molecule_search.mol_adapter import MolAdapter
from examples.molecule_search.mol_transformer.transformer import create_masks, Transformer, EXTRA_CHARS, ALPHABET_SIZE
from examples.molecule_search.utils import download_from_github
from golem.core.log import default_log
from golem.core.paths import project_root
from golem.core.utilities.data_structures import ensure_wrapped_in_sequence


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
    def __call__(self, obs):
        molecule = obs.get_rw_molecule()
        sentence = MolSentence(mol2alt_sentence(molecule, radius=1))
        embedding = self.sentences2vec([sentence], self.model)[0]
        embedding = ensure_wrapped_in_sequence(embedding)
        if len(embedding) < 300:
            embedding = embedding * 300
        return np.array(embedding).astype(float)

    def load_pretrained(self):
        save_dir = os.path.dirname(self.file_path)
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(self.file_path):
            self.log.message("Downloading pretrained model for molecules encoding...")
            response = requests.get(Mol2Vec.GITHUB_URL)
            with open(self.file_path, "wb") as new_file:
                new_file.write(response.content)

    @staticmethod
    def sentences2vec(sentences, model, unseen=None):
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

    def __init__(self, embedding_size=512, num_layers=6, max_length=256):
        self.log = default_log(self)

        self.file_path = os.path.join(project_root(), MoleculeTransformer.PRETRAINED_TRANSFORMER)
        download_from_github(self.file_path,
                             MoleculeTransformer.GITHUB_URL,
                             message="Downloading pretrained model for molecules encoding...")

        model = Transformer(ALPHABET_SIZE, embedding_size, num_layers).eval()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(self.file_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])

        self.model = model.module.cpu()
        self.encoder = self.model.encoder.cpu()
        self.max_length = max_length

    @adapter_method_to_molgraph
    def __call__(self, obs):
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

    def encode_smiles(self, string, start_char=EXTRA_CHARS['seq_start']):
        return torch.tensor([ord(start_char)]
                            + [self.encode_char(c) for c in string], dtype=torch.long)[:self.max_length].unsqueeze(0)
