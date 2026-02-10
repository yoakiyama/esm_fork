import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
from pathlib import Path

class trRosettaContactMSADataset(Dataset):
    def __init__(
        self,
        paired_paths_l: List[Tuple[Union[str, Path], Union[str, Path]]],
    ):
        self.paired_paths_l = paired_paths_l

    def __len__(self):
        return len(self.paired_paths_l)

    def __getitem__(self, idx):
        # Get MSA file path
        msa_path, npz_file_path = self.paired_paths_l[idx]
        # Get query sequence from msa_path
        with open(msa_path, 'r') as f:
            f.readline()
            query_sequence = f.readline().strip()
        res_d = {}
        res_d['sequence'] = query_sequence
        # Get contact map
        npz_obj = np.load(npz_file_path)
        res_d['contact_map'] = torch.tensor((npz_obj['dist6d'] > 0) & (npz_obj['dist6d'] < 8))
        # Add file path
        res_d['msa_file_path'] = msa_path
        res_d['npz_file_path'] = npz_file_path
        return res_d