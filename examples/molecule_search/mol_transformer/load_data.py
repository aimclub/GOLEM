import torch
from torch.utils.data import Dataset, DataLoader

PRINTABLE_ASCII_CHARS = 95

_extra_chars = ["seq_start", "seq_end", "pad"]
EXTRA_CHARS = {key: chr(PRINTABLE_ASCII_CHARS + i) for i, key in enumerate(_extra_chars)}
ALPHABET_SIZE = PRINTABLE_ASCII_CHARS + len(EXTRA_CHARS)


def encode_char(c):
    return ord(c) - 32


def decode_char(n):
    return chr(n + 32)


def smiles_iupac_batch(instances):
    smiles_lens = torch.tensor([s[0].shape[0] + 1 for s in instances], dtype=torch.long)
    iupac_lens = torch.tensor([s[1].shape[0] + 1 for s in instances], dtype=torch.long)
    
    max_len_smiles = smiles_lens.max().item()
    max_len_iupac = iupac_lens.max().item()
    
    batch_smiles = torch.full((len(instances), max_len_smiles), ord(EXTRA_CHARS['pad']), dtype=torch.long)
    batch_iupac_in = torch.full((len(instances), max_len_iupac), ord(EXTRA_CHARS['pad']), dtype=torch.long)
    batch_iupac_out = torch.full((len(instances), max_len_iupac), ord(EXTRA_CHARS['pad']), dtype=torch.long)

    for i, instance in enumerate(instances):
        batch_smiles[i, 0] = ord(EXTRA_CHARS['seq_start'])
        batch_smiles[i, 1:smiles_lens[i]] = instance[0]

        batch_iupac_in[i, 0] = ord(EXTRA_CHARS['seq_start'])
        batch_iupac_in[i, 1:iupac_lens[i]] = instance[1]

        batch_iupac_out[i, iupac_lens[i]-1] = ord(EXTRA_CHARS['seq_end'])
        batch_iupac_out[i, 0:iupac_lens[i]-1] = instance[1]
    
    return batch_smiles, batch_iupac_in, batch_iupac_out, smiles_lens, iupac_lens


class SmilesIupacDataset(Dataset):
    def __init__(self, data_path, max_len=None):
        self.pairs = [line.strip("\n").split("\t") for line in open(data_path, "r")]
        self.max_len = max_len - 1 if max_len else 0
    
    def __len__(self):
        return len(self.pairs)
    
    def string_to_tensor(self, string):
        tensor = torch.tensor(list(map(encode_char, string)), dtype=torch.uint8)
        
        if self.max_len > 0:
            tensor = tensor[:self.max_len]
        
        return tensor
    
    def __getitem__(self, index):
        smiles, iupac = self.pairs[index]
        return self.string_to_tensor(smiles), self.string_to_tensor(iupac)


def get_dataloader(batch_size, data_path, max_len=256):
    dataset = SmilesIupacDataset(data_path, max_len=max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=smiles_iupac_batch), dataset
