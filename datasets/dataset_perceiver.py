import os
import os.path as osp
import sys

import numpy as np

import torch

from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.nn import MessagePassing,  radius_graph, knn_graph
from torch_geometric.utils import coalesce, to_dense_adj,dense_to_sparse
from torch_geometric.datasets import QM9


class CustomDataset(QM9):

    def __init__(self, root: str, transform=None,
                 pre_transform=None,
                 pre_filter=None,
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

    def f(self, indexes, edge_index):
        helper_list = np.arange(len(indexes))
        dicti = {x: y for x, y in zip(helper_list, indexes.numpy())}
        return edge_index.apply_(lambda x: dicti[x])

    def deg(self, data_index, positions):
        angles = []
        for source, (leaf1, leaf2) in enumerate(data_index[0].reshape(-1, 2)):
            leaf1_sour = positions[leaf1] - positions[source]
            leaf2_sour = positions[leaf2] - positions[source]
            angles.append(torch.dot(leaf1_sour, leaf2_sour) / (torch.norm(leaf1_sour) * torch.norm(leaf2_sour)))
        return angles

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if True:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            atom_charges = [x['z'].apply_(lambda f: self.corr_dict[f]) for x in data_list]

            distances = []
            connections_rows = []
            connections_cols = []
            angles = []
            bonds = []

            for x in data_list:
                complete = radius_graph(x['pos'], r=10, batch=None, max_num_neighbors=32)
                adj_mat_complete = to_dense_adj(complete).type(torch.int8)
                adj_mat_bonds = to_dense_adj(x['edge_index'], edge_attr=torch.argmax(x['edge_attr'], dim=1) + 2)

                decreased, bond_feat = dense_to_sparse(torch.tril(
                    adj_mat_bonds | adj_mat_complete))  # torch.stack([x[0] for x in sorted([(x,y) for x,y in zip(complete.T, summed)], key = lambda x:( x[1], x[0][1]))], dim=0)[::2].T
                (row, col), pos = decreased, x['pos']
                dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
                distances.append(dist.reshape(-1))
                knn_edge_index = knn_graph(x['pos'], 2)
                angles.append(torch.hstack(self.deg(knn_edge_index, x['pos'])))
                x['edge_index'] = self.f(x['z'], decreased)
                (row, col) = x['edge_index']
                connections_rows.append(row)
                connections_cols.append(col)
                bonds.append(bond_feat)

            # print(len(bonds))
            # atom_charges = [x.apply_(lambda x: self.corr_dict[x]) for x in atom_charges]
            for x, y, dist, row, col, angle, bond in zip(data_list,
                                                         torch.nn.utils.rnn.pad_sequence(atom_charges,
                                                                                         batch_first=True), \
                                                         torch.nn.utils.rnn.pad_sequence(distances, batch_first=True), \
                                                         torch.nn.utils.rnn.pad_sequence(connections_rows,
                                                                                         batch_first=True), \
                                                         torch.nn.utils.rnn.pad_sequence(connections_cols,
                                                                                         batch_first=True), \
                                                         torch.nn.utils.rnn.pad_sequence(angles, batch_first=True,
                                                                                         padding_value=10), \
                                                         torch.nn.utils.rnn.pad_sequence(bonds, batch_first=True)):
                x['modif_z'] = y
                x['attent_mask'] = (x['modif_z'] != 0).type(torch.int)
                x['distances_padded'] = dist
                x['attent_dist'] = (x['distances_padded'] != 0).type(torch.int)
                x['row_padded'] = row
                x['col_padded'] = col
                x['angle_padded'] = angle
                x['bond'] = bond

            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            # distances = [x.e]
            torch.save(self.collate(data_list), self.processed_paths[0])
            return

# class CustomDataset(QM9):
#
#     def __init__(self, root: str, transform=None,
#                  pre_transform=None,
#                  pre_filter=None,
#                  corr_dict={1: 1, 6: 2, 7: 3, 8: 4, 9: 5}):
#         self.corr_dict = corr_dict
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     def download(self):
#         try:
#             import rdkit  # noqa
#             file_path = download_url(self.raw_url, self.raw_dir)
#             extract_zip(file_path, self.raw_dir)
#             os.unlink(file_path)
#
#             file_path = download_url(self.raw_url2, self.raw_dir)
#             os.rename(osp.join(self.raw_dir, '3195404'),
#                       osp.join(self.raw_dir, 'uncharacterized.txt'))
#         except ImportError:
#             path = download_url(self.processed_url, self.raw_dir)
#             extract_zip(path, self.raw_dir)
#             os.unlink(path)
#
#     def f(self, indexes, edge_index):
#         helper_list = np.arange(len(indexes))
#         dicti = {x: y for x, y in zip(helper_list, indexes.numpy())}
#         return edge_index.apply_(lambda x: dicti[x])
#
#     def deg(self, data_index, positions):
#         angles = []
#         for source, (leaf1, leaf2) in enumerate(data_index[0].reshape(-1, 2)):
#             leaf1_sour = positions[leaf1] - positions[source]
#             leaf2_sour = positions[leaf2] - positions[source]
#             angles.append(torch.dot(leaf1_sour, leaf2_sour) / (torch.norm(leaf1_sour) * torch.norm(leaf2_sour)))
#         return angles
#
#     def process(self):
#         try:
#             import rdkit
#             from rdkit import Chem, RDLogger
#             from rdkit.Chem.rdchem import BondType as BT
#             from rdkit.Chem.rdchem import HybridizationType
#             RDLogger.DisableLog('rdApp.*')
#
#         except ImportError:
#             rdkit = None
#
#         if rdkit is None:
#             print(("Using a pre-processed version of the dataset. Please "
#                    "install 'rdkit' to alternatively process the raw data."),
#                   file=sys.stderr)
#
#             data_list = torch.load(self.raw_paths[0])
#             atom_charges = [x['z'].apply_(lambda f: self.corr_dict[f]) for x in data_list]
#
#             distances = []
#             connections_rows = []
#             connections_cols = []
#             angles = []
#
#             for x in data_list:
#                 complete = radius_graph(x['pos'], r=10, batch=None, max_num_neighbors=32)
#                 decreased = dense_to_sparse(torch.tril(to_dense_adj(complete)))[
#                     0]  # torch.stack([x[0] for x in sorted([(x,y) for x,y in zip(complete.T, summed)], key = lambda x:( x[1], x[0][1]))], dim=0)[::2].T
#                 (row, col), pos = decreased, x['pos']
#                 dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
#                 distances.append(dist.reshape(-1))
#                 knn_edge_index = knn_graph(x['pos'], 2)
#                 angles.append(torch.hstack(self.deg(knn_edge_index, x['pos'])))
#                 x['edge_index'] = self.f(x['z'], decreased)
#                 (row, col) = x['edge_index']
#                 connections_rows.append(row)
#                 connections_cols.append(col)
#
#             # atom_charges = [x.apply_(lambda x: self.corr_dict[x]) for x in atom_charges]
#             for x, y, dist, row, col, angle in zip(data_list,
#                                                    torch.nn.utils.rnn.pad_sequence(atom_charges, batch_first=True), \
#                                                    torch.nn.utils.rnn.pad_sequence(distances, batch_first=True), \
#                                                    torch.nn.utils.rnn.pad_sequence(connections_rows, batch_first=True), \
#                                                    torch.nn.utils.rnn.pad_sequence(connections_cols, batch_first=True), \
#                                                    torch.nn.utils.rnn.pad_sequence(angles, batch_first=True,
#                                                                                    padding_value=10)):
#                 x['modif_z'] = y
#                 x['attent_mask'] = (x['modif_z'] != 0).type(torch.int)
#                 x['distances_padded'] = dist
#                 x['attent_dist'] = (x['distances_padded'] != 0).type(torch.int)
#                 x['row_padded'] = row
#                 x['col_padded'] = col
#                 x['angle_padded'] = angle
#
#             data_list = [Data(**data_dict) for data_dict in data_list]
#
#             if self.pre_filter is not None:
#                 data_list = [d for d in data_list if self.pre_filter(d)]
#
#             if self.pre_transform is not None:
#                 data_list = [self.pre_transform(d) for d in data_list]
#
#             # distances = [x.e]
#             torch.save(self.collate(data_list), self.processed_paths[0])
#             return

