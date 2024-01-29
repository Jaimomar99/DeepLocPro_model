from typing import Any, Sequence, Tuple, List

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import pandas as pd
from hashlib import md5
import numpy as np

# we do not use this because the data is 0 or 1 by column if it's in that location


KINGDOM_TO_INT = {"ARCHAEA":0,
                    "POSITIVE":1,
                    "NEGATIVE":2}


def make_hashes(names: List[str]) -> List[str]:
    hashes = []
    for name in names:
        hashes.append(md5(name.encode()).digest().hex())

    return hashes


class PrecomputedCSVDataset(Dataset):
    '''Use together with modified extract.py script. Retrieves seqs via md5 hash.'''
    def __init__(
        self, 
        embeddings_dir: str, 
        data_file: str, 
        partitioning_file: str, 
        partitions: List[int]=[0], 
        label_type: str = 'binary',
        ):
        """
        Dataset to hold fasta files with precomputed embeddings.
        Can also parse graph-part partition assignments.
        Args:
            embeddings_dir (str): Directory containing precomputed embeddings produced by `extract.py`
            csv_file (str): csv with sequences, labels and other metadata.
            partitioning_file (str): Graph-Part output for `data_file`. Defaults to None.
            partitions (List[int], optional): Partitions to retain. Defaults to [0].
        """

        super().__init__()
        self.embeddings_dir = embeddings_dir
        data = pd.read_csv(data_file, index_col='Entry')        
        partitioning = pd.read_csv(partitioning_file, index_col='AC')
        data = data.join(partitioning)
        data = data.loc[data['cluster'].isin(partitions)]
        self.data = data
        

        self.names = data.index.tolist() # don't want to bother with pandas indexing here.
        self.sequences = data['Sequence'].tolist()
        self.organism = data['Kingdom'].tolist()
        self.hashes = make_hashes(self.sequences)

        labels = data['Cellwall'].astype(str) + data['Extracellular'].astype(str)+ data['Cytoplasmic'].astype(str)+ data['CYtoplasmicMembrane'].astype(str)+ data['OuterMembrane'].astype(str)+ data['Periplasmic'].astype(str)
        self.labels = [[int(j) for j in i] for i in labels.tolist()]
        
    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        seq_hash = self.hashes[index]
        try:
            embeddings = torch.load(os.path.join(self.embeddings_dir, f'{seq_hash}.pt'))
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find sequence hash {seq_hash} for {self.names[index]} in {self.embeddings_dir}.')
        
        #LABEL_TO_INT = dict([(label,idx) for idx, label in enumerate(set(self.labels))])
        label = self.labels[index]
        # mask : batch_size, seq_len
        mask = torch.ones(embeddings.shape[0])
        kingdom = KINGDOM_TO_INT[self.organism[index]]
        return embeddings, mask, label, kingdom


    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        For each batch, it allows all the sequences have the same length padding the short ones.
        '''
        embeddings, masks, labels, kingdoms = zip(*batch)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
        labels = torch.LongTensor(labels)
        kingdoms = torch.LongTensor(kingdoms)
        
        return embeddings, masks, labels, kingdoms