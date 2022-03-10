import os
import os.path as osp
import sys
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.datasets import QM9


class CustomDataset(QM9):

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 corr_dict={1: 1, 6: 2, 7: 3, 8: 4, 9: 5}):
        self.corr_dict = corr_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            atom_charges = [x['z'].apply_(lambda f: self.corr_dict[f]) for x in data_list]

            # atom_charges = [x.apply_(lambda x: self.corr_dict[x]) for x in atom_charges]
            for x, y in zip(data_list, torch.nn.utils.rnn.pad_sequence(atom_charges, batch_first=True)):
                x['modif_z'] = y
                x['attent_mask'] = (x['modif_z'] != 0).type(torch.int)
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return